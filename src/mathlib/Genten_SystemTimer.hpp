//@HEADER
// ************************************************************************
//     Genten: Software for Generalized Tensor Decompositions
//     by Sandia National Laboratories
//
// Sandia National Laboratories is a multimission laboratory managed
// and operated by National Technology and Engineering Solutions of Sandia,
// LLC, a wholly owned subsidiary of Honeywell International, Inc., for the
// U.S. Department of Energy's National Nuclear Security Administration under
// contract DE-NA0003525.
//
// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
// Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// ************************************************************************
//@HEADER

/*!
  @file Genten_SystemTimer.h
  @brief Class declaration of Genten::SystemTimer (adapted from HOPSPACK).
*/

#ifndef GENTEN_SYSTEMTIMER_H
#define GENTEN_SYSTEMTIMER_H

#include <string>

#include "Genten_Pmap.hpp"

namespace Genten
{


//----------------------------------------------------------------------
//! System-dependent timer for wall clock time and date.
/*!
 *  On Windows the wall clock timer has resolution of 1 millisecond,
 *  on Unix the resolution is 1 microsecond.  The CPU time required to
 *  query a timer is about 100 nanoseconds on Windows, 1.5 microseconds
 *  on Linux (of course it depends on the processor).
 *  Constructing a timer object takes 35 milliseconds on Windows, so
 *  applications should reuse a timer rather than constantly recreate one.
 *
 *  Example use:
 *    SystemTimer  timers (5);     //-- DEFINES TIMERS 0..4
 *    timers.start (0);
 *    ...
 *    timers.stop (0);
 *    double  dTimeInSec = timers.getTotalTime (0);
 */
//----------------------------------------------------------------------
  class SystemTimer
  {
  public:

    //! Constructor.
    /*!
     *  @param[in] nNumTimers  Number of timers to allocate.
     *  @param[in] fence       Call Kokkos::fence() when stopping timer
     */
    SystemTimer (const int  nNumTimers = 0,
                 const bool fence = false,
                 const ProcessorMap *pmap = nullptr);

    //! Destructor.
    ~SystemTimer ();

    //! Initialize.
    /*!
     *  @param[in] nNumTimers  Number of timers to allocate.
     *  @param[in] fence       Call Kokkos::fence() when stopping timer
     */
    void init (const int nNumTimers, const bool fence,
               const ProcessorMap *pmap = nullptr);

    //! Destroy
    void destroy();

    //! Static method to return current date and time.
    /*!
     *  @param[out]  sCurrentDateTime  Format will be MM/DD/YYYY HH:MM:SS.
     */
    static void  getDateTime (std::string &  sCurrentDateTime);

    //! Static method to put the current thread to sleep.
    /*!
     *  @param[in] nMilliSecs  Number of milliseconds to sleep.  If the number
     *                         is zero or negative, then no sleep occurs.
     *                         Exact sleep time depends on the operating system,
     *                         but is typically very close to the target.
     */
    static void  sleepMilliSecs (const int  nMilliSecs);

    //! Start a particular timer.  Should not be called twice in a row.
    /*!
     *  @param[in] nTimerID  The timer of interest (>= 0).
     *  @return              True if successful.
     */
    bool  start (const int  nTimerID);

    //! Stop a particular timer.  Should not be called twice in a row.
    /*!
     *  @param[in] nTimerID  The timer of interest (>= 0).
     *  @return              True if successful.
     */
    bool  stop (const int  nTimerID);

    //! Return the total "start" to "stop" duration for the timer.
    /*!
     *  @param[in] nTimerID  The timer of interest (>= 0).
     *  @return              Total elapsed time between all calls to start()
     *                       and stop() since the class instance was constructed,
     *                       or since reset() was called.  If there is currently
     *                       a start() with no matching stop(), then the total
     *                       also includes all time since the last start().
     */
    double  getTotalTime (const int  nTimerID) const;

    //! Return the number of times start() was called for the timer.
    int  getNumStarts (const int  nTimerID) const;

    //! Return the average "start" to "stop" duration for the timer.
    /*!
     *  @param[in] nTimerID  The timer of interest (>= 0).
     *  @return              Average elapsed time between all calls to start()
     *                       and stop() since the class instance was constructed,
     *                       or since reset() was called.  Unlike getTotalTime(),
     *                       if there is currently a start() with no matching
     *                       stop(), then the total ignores time since this
     *                       last start().
     *
     *  Result is the same as computing getTotalTime (n) / getNumStarts (n).
     */
    double  getAvgTime (const int  nTimerID) const;

    //! Reset the timer so it can be started again from zero.
    void  reset (const int  nTimerID);


  private:

    //! By design, there is no copy constructor.
    SystemTimer (const SystemTimer &);
    //! By design, there is no assignment operator.
    SystemTimer & operator= (const SystemTimer &);

    //! Return the time in seconds since the last call to start().
    double  getTimeSinceLastStart_ (const int  nTimerID) const;


    //! Hide operating system details in this private type.
    typedef struct _systemTimerRealType _systemTimerRealType;
    _systemTimerRealType *  _taStartTimes;

    int       _nNumTimers;
    bool      _fence;
    const ProcessorMap *_pmap;
    bool *    _baIsStarted;
    double *  _daCumTimes;
    int *     _naNumCalls;

#if defined(_WIN32)
    int       _nWindowsSaveTimerMin;
#endif
  };

}          //-- namespace Genten

#endif     //-- GENTEN_SYSTEMTIMER_H
