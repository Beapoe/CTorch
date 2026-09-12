#include "../../include/DeviceAllocator.h"
#include "../../include/MetalDevice.h"
#include <unordered_map>
#include <mutex>

struct MPSAllocatorImpl {
    std::unordered_map<void*, id<MTLBuffer>> bufferMap;
    std::mutex mapMutex;
};

MPSAllocator::MPSAllocator() {
    _impl = new MPSAllocatorImpl();
}

MPSAllocator::~MPSAllocator() {
    delete static_cast<MPSAllocatorImpl*>(_impl);
}

void* MPSAllocator::allocate(size_t bytes, DeviceType device) {
    if (device != DeviceType::kMPS) {
        return nullptr;
    }
    id<MTLBuffer> buffer = MetalDevice::getInstance().allocateBuffer(bytes, MTLResourceStorageModeShared);
    void* ptr = [buffer contents];
    {
        std::lock_guard<std::mutex> lock(static_cast<MPSAllocatorImpl*>(_impl)->mapMutex);
        static_cast<MPSAllocatorImpl*>(_impl)->bufferMap[ptr] = buffer;
    }
    return ptr;
}

void MPSAllocator::deallocate(void* ptr, DeviceType device) {
    if (device != DeviceType::kMPS || ptr == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(static_cast<MPSAllocatorImpl*>(_impl)->mapMutex);
    static_cast<MPSAllocatorImpl*>(_impl)->bufferMap.erase(ptr);
}

void MPSAllocator::memcpy(void* dst, const void* src, size_t bytes,
                          DeviceType dst_device, DeviceType src_device) {
    std::memcpy(dst, src, bytes);
}

extern "C" id<MTLBuffer> MPS_getBuffer(void* ptr) {
    std::shared_ptr<DeviceAllocator> allocator =
        AllocatorManager::getInstance().getAllocator(DeviceType::kMPS);
    if (!allocator || ptr == nullptr) {
        return nil;
    }
    MPSAllocator* mps_alloc = static_cast<MPSAllocator*>(allocator.get());
    MPSAllocatorImpl* impl = static_cast<MPSAllocatorImpl*>(mps_alloc->getImpl());
    std::lock_guard<std::mutex> lock(impl->mapMutex);

    auto it = impl->bufferMap.find(ptr);
    if (it != impl->bufferMap.end()) {
        return it->second;
    }

    // Fallback: ptr may point inside a buffer (e.g., a view with storage_offset > 0).
    // BufferMap keys are base pointers returned by [buffer contents], so scan for the
    // containing buffer. This is O(n) but only runs on lookup miss.
    for (auto& pair : impl->bufferMap) {
        void* base = pair.first;
        id<MTLBuffer> buffer = pair.second;
        // [Fix §4.95 P2] 无关系 void* 的 < 比较为 UB, 改 uintptr_t 整数比较
        uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t b = reinterpret_cast<uintptr_t>(base);
        if (p >= b && p < b + [buffer length]) {
            return buffer;
        }
    }

    static std::once_flag print_once;
    std::call_once(print_once, [&]() {
        std::cout << "[DEBUG] MPS_getBuffer: ptr=" << ptr << " not found! map size=" << impl->bufferMap.size() << std::endl;
        for (auto& pair : impl->bufferMap) {
            std::cout << "[DEBUG] MPS_getBuffer: key=" << pair.first << std::endl;
        }
    });

    return nil;
}

extern "C" void MPS_markBufferModified(void* ptr, size_t bytes) {
    std::shared_ptr<DeviceAllocator> allocator =
        AllocatorManager::getInstance().getAllocator(DeviceType::kMPS);
    if (!allocator || ptr == nullptr || bytes == 0) {
        return;
    }
    MPSAllocator* mps_alloc = static_cast<MPSAllocator*>(allocator.get());
    MPSAllocatorImpl* impl = static_cast<MPSAllocatorImpl*>(mps_alloc->getImpl());
    std::lock_guard<std::mutex> lock(impl->mapMutex);

    auto it = impl->bufferMap.find(ptr);
    if (it != impl->bufferMap.end()) {
        [it->second didModifyRange:NSMakeRange(0, bytes)];
        return;
    }

    // Fallback: ptr may point inside a buffer (e.g., a view with storage_offset > 0).
    for (auto& pair : impl->bufferMap) {
        void* base = pair.first;
        id<MTLBuffer> buffer = pair.second;
        // [Fix §4.95 P2] 无关系 void* 的 < 比较为 UB, 改 uintptr_t 整数比较
        uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t b = reinterpret_cast<uintptr_t>(base);
        if (p >= b && p < b + [buffer length]) {
            size_t offset = static_cast<char*>(ptr) - static_cast<char*>(base);
            [buffer didModifyRange:NSMakeRange(offset, bytes)];
            return;
        }
    }
}