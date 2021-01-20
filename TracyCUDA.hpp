#ifndef __TRACYCUDA_HPP__
#define __TRACYCUDA_HPP__

#if !defined TRACY_ENABLE

#define TracyCUDAContext(c, x) nullptr
#define TracyCUDADestroy(c)

#define TracyCUDANamedZone(c, x, y, z)
#define TracyCUDANamedZoneC(c, x, y, z, w)
#define TracyCUDAZone(c, x)
#define TracyCUDAZoneC(c, x, y)

#define TracyCUDANamedZoneS(c, x, y, z, w)
#define TracyCUDANamedZoneCS(c, x, y, z, w, v)
#define TracyCUDAZoneS(c, x, y)
#define TracyCUDAZoneCS(c, x, y, z)

#define TracyCUDANamedZoneSetEvent(x, e)
#define TracyCUDAZoneSetEvent(e)

#define TracyCUDACollect(c)

namespace tracy
{
    class CUDACtxScope {};
}

using TracyCUDACtx = void*;

#else

#include <cuda.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cassert>

#include "Tracy.hpp"
#include "client/TracyCallstack.hpp"
#include "client/TracyProfiler.hpp"
#include "common/TracyAlloc.hpp"
#include "common/TracyQueue.hpp"

namespace tracy {

    struct synched_stamp {
        uint64_t cpu;
        cudaEvent_t gpu;
    };

    class CUDACtx
    {
    public:
        static constexpr int maxEvents = 64 * 1024;

    private:
        unsigned int m_contextId;

        cudaEvent_t m_events[maxEvents];
        unsigned int m_head;
        unsigned int m_tail;

        synched_stamp m_startTime;

    public:

        CUDACtx()
            : m_contextId(GetGpuCtxCounter().fetch_add(1, std::memory_order_relaxed))
            , m_head(0)
            , m_tail(0)
        {
            assert(m_contextId != 255);

            for (auto i=0; i<maxEvents; ++i) {
                assert(CUDA_SUCCESS == cudaEventCreate(m_events+i));
            }

            // Set synchronized time stamp
            setSynchedBaseTime();

            // Push context
            auto item = Profiler::QueueSerial();
            MemWrite(&item->hdr.type, QueueType::GpuNewContext);
            MemWrite(&item->gpuNewContext.cpuTime, m_startTime.cpu);
            MemWrite(&item->gpuNewContext.gpuTime, m_startTime.cpu);
            memset(&item->gpuNewContext.thread, 0, sizeof(item->gpuNewContext.thread));
            MemWrite(&item->gpuNewContext.period, 1.0f);
            MemWrite(&item->gpuNewContext.type, GpuContextType::CUDA);
            MemWrite(&item->gpuNewContext.context, (uint8_t) m_contextId);
            MemWrite(&item->gpuNewContext.flags, (uint8_t)0);
#ifdef TRACY_ON_DEMAND
            GetProfiler().DeferItem(*item);
#endif
            Profiler::QueueSerialFinish();
        }

        ~CUDACtx()
        {
            for (auto i=0; i<maxEvents; ++i)
            {
                assert(CUDA_SUCCESS == cudaEventDestroy(m_events[i]));
            }
        }

        void Collect()
        {
            ZoneScopedC(Color::Red4);

            if (m_tail == m_head) return;

#ifdef TRACY_ON_DEMAND
            if (!GetProfiler().IsConnected())
            {
                m_head = m_tail = 0;
            }
#endif

            while (m_tail != m_head)
            {
                uint64_t gpuTime = timeSince(m_events[m_tail]);
                auto item = Profiler::QueueSerial();
                MemWrite(&item->hdr.type, QueueType::GpuTime);
                MemWrite(&item->gpuTime.gpuTime, gpuTime);
                MemWrite(&item->gpuTime.queryId, (uint16_t)m_tail);
                MemWrite(&item->gpuTime.context, m_contextId);
                Profiler::QueueSerialFinish();

                m_tail = (m_tail + 1) % maxEvents;
            }
        }

        tracy_force_inline unsigned int nextQueryId()
        {
            const auto id = m_head;
            m_head = ( m_head + 1 ) % maxEvents;
            assert( m_head != m_tail );
            return id;
        }

        tracy_force_inline uint8_t contextId() const
        {
            return m_contextId;
        }

    private:

        tracy_force_inline void setSynchedBaseTime()
        {
            // Perform a dummy ping pong copy, which will enqueue two copies in
            // the stream. Then insert a CUDA event after the copies and
            // synchronize the host and device, to take a cpu timestamp
            // synchronized with the event.
            int* devPtr;
            cudaMalloc(&devPtr, sizeof(int));
            int value = 42;
            cudaMemcpy(devPtr, &value, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(&value, devPtr, sizeof(int), cudaMemcpyDeviceToHost);

            cudaEventRecord(m_startTime.gpu);

            cudaDeviceSynchronize();

            m_startTime.cpu = Profiler::GetTime();

            cudaFree(devPtr);
        }

        tracy_force_inline uint64_t timeSince(const cudaEvent_t e)
        {
            float time_taken = 0.0f;
            assert(CUDA_SUCCESS == cudaEventElapsedTime(&time_taken, m_startTime.gpu, e));
            return m_startTime.cpu + static_cast<uint64_t>(time_taken*1.e6);
        }
    };

    class CUDACtxScope {
    public:
        tracy_force_inline CUDACtxScope(CUDACtx* ctx, const SourceLocationData* srcLoc, bool is_active)
#ifdef TRACY_ON_DEMAND
            : m_active(is_active&& GetProfiler().IsConnected())
#else
            : m_active(is_active)
#endif
            , m_ctx(ctx)
        {
            if (!m_active) return;

            m_beginQueryId = ctx->nextQueryId();

            auto item = Profiler::QueueSerial();
            MemWrite(&item->hdr.type, QueueType::GpuZoneBeginSerial);
            MemWrite(&item->gpuZoneBegin.cpuTime, Profiler::GetTime());
            MemWrite(&item->gpuZoneBegin.srcloc, (uint64_t)srcLoc);
            MemWrite(&item->gpuZoneBegin.thread, GetThreadHandle());
            MemWrite(&item->gpuZoneBegin.queryId, (uint16_t)m_beginQueryId);
            MemWrite(&item->gpuZoneBegin.context, ctx->contextId());
            Profiler::QueueSerialFinish();
        }

        tracy_force_inline CUDACtxScope(CUDACtx* ctx, const SourceLocationData* srcLoc, int depth, bool is_active)
#ifdef TRACY_ON_DEMAND
            : m_active(is_active&& GetProfiler().IsConnected())
#else
            : m_active(is_active)
#endif
            , m_ctx(ctx)
        {
            if (!m_active) return;

            m_beginQueryId = ctx->nextQueryId();

            GetProfiler().SendCallstack(depth);

            auto item = Profiler::QueueSerial();
            MemWrite(&item->hdr.type, QueueType::GpuZoneBeginCallstackSerial);
            MemWrite(&item->gpuZoneBegin.cpuTime, Profiler::GetTime());
            MemWrite(&item->gpuZoneBegin.srcloc, (uint64_t)srcLoc);
            MemWrite(&item->gpuZoneBegin.thread, GetThreadHandle());
            MemWrite(&item->gpuZoneBegin.queryId, (uint16_t)m_beginQueryId);
            MemWrite(&item->gpuZoneBegin.context, ctx->contextId());
            Profiler::QueueSerialFinish();
        }

        tracy_force_inline ~CUDACtxScope()
        {
            const auto queryId = m_ctx->nextQueryId();

            auto item = Profiler::QueueSerial();
            MemWrite(&item->hdr.type, QueueType::GpuZoneEndSerial);
            MemWrite(&item->gpuZoneEnd.cpuTime, Profiler::GetTime());
            MemWrite(&item->gpuZoneEnd.thread, GetThreadHandle());
            MemWrite(&item->gpuZoneEnd.queryId, (uint16_t)queryId);
            MemWrite(&item->gpuZoneEnd.context, m_ctx->contextId());
            Profiler::QueueSerialFinish();
        }

        const bool m_active;
        CUDACtx* m_ctx;
        unsigned int m_beginQueryId;
    };

    static inline CUDACtx* CreateCUDAContext() // TODO: make this consume cuda context/cuda stream/device? and forward it.
    {
        InitRPMallocThread();
        auto ctx = (CUDACtx*)tracy_malloc(sizeof(CUDACtx));
        new (ctx) CUDACtx();
        return ctx;
    }

    static inline void DestroyCUDAContext(CUDACtx* ctx)
    {
        ctx->~CUDACtx();
        tracy_free(ctx);
    }

}  // namespace tracy

/*
using TracyCLCtx = tracy::OpenCLCtx*;

#define TracyCLContext(context, device) tracy::CreateCLContext(context, device);
#define TracyCLDestroy(ctx) tracy::DestroyCLContext(ctx);
#if defined TRACY_HAS_CALLSTACK && defined TRACY_CALLSTACK
#  define TracyCLNamedZone(ctx, varname, name, active) static constexpr tracy::SourceLocationData TracyConcat(__tracy_gpu_source_location,__LINE__) { name, __FUNCTION__, __FILE__, (uint32_t)__LINE__, 0 }; tracy::OpenCLCtxScope varname(ctx, &TracyConcat(__tracy_gpu_source_location,__LINE__), TRACY_CALLSTACK, active );
#  define TracyCLNamedZoneC(ctx, varname, name, color, active) static constexpr tracy::SourceLocationData TracyConcat(__tracy_gpu_source_location,__LINE__) { name, __FUNCTION__, __FILE__, (uint32_t)__LINE__, color }; tracy::OpenCLCtxScope varname(ctx, &TracyConcat(__tracy_gpu_source_location,__LINE__), TRACY_CALLSTACK, active );
#  define TracyCLZone(ctx, name) TracyCLNamedZoneS(ctx, __tracy_gpu_zone, name, TRACY_CALLSTACK, true)
#  define TracyCLZoneC(ctx, name, color) TracyCLNamedZoneCS(ctx, __tracy_gpu_zone, name, color, TRACY_CALLSTACK, true)
#else
#  define TracyCLNamedZone(ctx, varname, name, active) static constexpr tracy::SourceLocationData TracyConcat(__tracy_gpu_source_location,__LINE__){ name, __FUNCTION__, __FILE__, (uint32_t)__LINE__, 0 }; tracy::OpenCLCtxScope varname(ctx, &TracyConcat(__tracy_gpu_source_location,__LINE__), active);
#  define TracyCLNamedZoneC(ctx, varname, name, color, active) static constexpr tracy::SourceLocationData TracyConcat(__tracy_gpu_source_location,__LINE__){ name, __FUNCTION__, __FILE__, (uint32_t)__LINE__, color }; tracy::OpenCLCtxScope varname(ctx, &TracyConcat(__tracy_gpu_source_location,__LINE__), active);
#  define TracyCLZone(ctx, name) TracyCLNamedZone(ctx, __tracy_gpu_zone, name, true)
#  define TracyCLZoneC(ctx, name, color) TracyCLNamedZoneC(ctx, __tracy_gpu_zone, name, color, true )
#endif

#ifdef TRACY_HAS_CALLSTACK
#  define TracyCLNamedZoneS(ctx, varname, name, depth, active) static constexpr tracy::SourceLocationData TracyConcat(__tracy_gpu_source_location,__LINE__){ name, __FUNCTION__, __FILE__, (uint32_t)__LINE__, 0 }; tracy::OpenCLCtxScope varname(ctx, &TracyConcat(__tracy_gpu_source_location,__LINE__), depth, active);
#  define TracyCLNamedZoneCS(ctx, varname, name, color, depth, active) static constexpr tracy::SourceLocationData TracyConcat(__tracy_gpu_source_location,__LINE__){ name, __FUNCTION__, __FILE__, (uint32_t)__LINE__, color }; tracy::OpenCLCtxScope varname(ctx, &TracyConcat(__tracy_gpu_source_location,__LINE__), depth, active);
#  define TracyCLZoneS(ctx, name, depth) TracyCLNamedZoneS(ctx, __tracy_gpu_zone, name, depth, true)
#  define TracyCLZoneCS(ctx, name, color, depth) TracyCLNamedZoneCS(ctx, __tracy_gpu_zone, name, color, depth, true)
#else
#  define TracyCLNamedZoneS(ctx, varname, name, depth, active) TracyCLNamedZone(ctx, varname, name, active)
#  define TracyCLNamedZoneCS(ctx, varname, name, color, depth, active) TracyCLNamedZoneC(ctx, varname, name, color, active)
#  define TracyCLZoneS(ctx, name, depth) TracyCLZone(ctx, name)
#  define TracyCLZoneCS(ctx, name, color, depth) TracyCLZoneC(ctx, name, color)
#endif

#define TracyCLNamedZoneSetEvent(varname, event) varname.SetEvent(event)
#define TracyCLZoneSetEvent(event) __tracy_gpu_zone.SetEvent(event)

#define TracyCLCollect(ctx) ctx->Collect()
*/

#endif

#endif
