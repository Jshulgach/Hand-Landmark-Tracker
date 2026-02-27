//======================================================================================================-----
// Copyright 2015, NaturalPoint Inc.
//======================================================================================================-----

//== simple threaded dispatcher for threaded tasks. ==--

#pragma once

#include <vector>

#include "Core/Event.h"
#include "Core/ThreadHandle.h"

namespace Core
{
    /// <summary>Override cThreadedTask to store pertinent information regarding
    ///    the threaded task to be performed.</summary>

    class cThreadingTask
    {
    public:
        cThreadingTask() {};
    };

    /// <summary>Inherit cThreadedDispatch to add threaded dispatching to your class.</summary>

    class cThreadedDispatch : public Core::cThreadProc
    {
    public:
        cThreadedDispatch();
        ~cThreadedDispatch();

        /// <summary>Dispatch a task to be scheduled and performed by one of the worker threads.</summary>
        void Dispatch( Core::cThreadingTask & task );

        /// <summary>Dispatch a task without threading. This is typically to facilitate debugging of threaded algorithms.</summary>
        void DispatchNoThreading( Core::cThreadingTask & task );

        /// <summary>Override PerformThreadedTask to receive tasks to perform. This
        ///    method will be called in parallel from a number of worker threads.</summary>
        virtual void PerformThreadedTask( Core::cThreadingTask & task );
        
        /// <summary>Parallelism defaults to the number of cores on the processor. Override
        ///   this to limit the number of task running in parallel.</summary>
        virtual int  MaximumThreadCount() const;

        /// <summary>Call WaitForThreadedCompletion() to block until all dispatched tasks are complete.</summary>
        void WaitForThreadedCompletion() const;

        /// <summary>Query IsAllThreadsIdle() to determine if all tasks are complete without blocking.</summary>
        bool IsAllThreadsIdle() const;

        /// <summary>ThreadProc method contains the default worker thread implementation for
        ///    performing tasks in parallel.  Needs to be public for thread entry point to call in.</summary>
        virtual void ThreadProc( Core::cThreadHandle & handle );

    private:
        void PrepareThreading();
        void ShutdownThreads();

        Core::cThreadHandle * AvailableThread();
        Core::cThreadHandle * NextAvailableThread();

        void Dispatch( Core::cThreadingTask * task ,Core::cThreadHandle * thread );

        std::vector<Core::cThreadHandle *> mThreadList;

        Core::cEvent mThreadedDispatcherSignal;

        int  mTargetThreadCount;
    };    
}


