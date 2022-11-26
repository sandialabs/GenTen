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
  @file Genten_SystemTimer.cpp
  @brief Implement Genten::SystemTimer (adapted from HOPSPACK).
*/

#if defined(_WIN32)
#include <windows.h>
#else
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#endif

#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "Genten_SystemTimer.hpp"
#include "Genten_Kokkos.hpp"


//---- THIS IS WHERE HAVE_REALTIME_CLOCK WILL BE DEFINED (OR NOT).
#include "CMakeInclude.h"


namespace Genten
{


//----------------------------------------------------------------------
//  Private structures
//----------------------------------------------------------------------

#if defined(HAVE_REALTIME_CLOCK)
//---- HIDE OPERATING SYSTEM DETAILS IN THESE STRUCTURES.
  struct _systemTimerRealType
  {
#if defined(_WIN32)
    DWORD  nnRealTime;      //-- PRIMITIVE RETURNED BY WINDOWS TIMER CALL
#else
    timeval  cRealTime;     //-- PRIMITIVE RETURNED BY UNIX TIMER CALL
#endif
  };
#endif


//----------------------------------------------------------------------
//  Constructor
//----------------------------------------------------------------------
  SystemTimer::SystemTimer (const int  nNumTimers,
                            const bool fence,
                            const ProcessorMap *pmap) : _nNumTimers(0), _pmap(pmap)
  {
    init(nNumTimers, fence, pmap);
  }

//----------------------------------------------------------------------
//  Destructor
//----------------------------------------------------------------------
  SystemTimer::~SystemTimer ()
  {
    destroy();
  }

//----------------------------------------------------------------------
//  Initialize
//----------------------------------------------------------------------
  void SystemTimer::init(const int  nNumTimers,
                         const bool fence,
                         const ProcessorMap *pmap)
  {
    // Destroy any previous initialization
    destroy();

    _pmap = pmap;

#if defined(HAVE_REALTIME_CLOCK)
    if (nNumTimers <= 0)
    {
      _nNumTimers = 0;
      return;
    }
    _nNumTimers = nNumTimers;
    _fence = fence;

    _baIsStarted = new bool[_nNumTimers];
    _daCumTimes = new double[_nNumTimers];
    _naNumCalls = new int[_nNumTimers];
    _taStartTimes = new _systemTimerRealType[_nNumTimers];

    for (int  i = 0; i < _nNumTimers; i++)
    {
      _baIsStarted[i] = false;
      _daCumTimes[i] = 0.0;
      _naNumCalls[i] = 0;
    }

    //---- NEED TO SET UP THE WINDOWS REAL TIME RESOLUTION TIMER.
#if defined(_WIN32)
    //---- MSDN SAYS TO SET THE RESOLUTION OF timeGetTime ONCE BEFORE
    //---- USING IT, AND REMEMBER TO UNSET WHEN FINISHED
    //---- (CALL timeBeginPeriod AND timeEndPeriod).
    //---- I FIND THAT IT TAKES timeBeginPeriod AT LEAST ONE SYSTEM TICK
    //---- (~16 MSEC) TO TAKE EFFECT; SLEEP TO SAFELY SET IT UP.
    _nWindowsSaveTimerMin = -1;
    TIMECAPS  tTimeCaps;
    if (timeGetDevCaps (&tTimeCaps,
                        sizeof (tTimeCaps)) != TIMERR_NOERROR)
    {
      _nWindowsSaveTimerMin = (int) tTimeCaps.wPeriodMin;
      timeBeginPeriod (_nWindowsSaveTimerMin);
      Sleep (35);     //-- 35 milliseconds
    }
    else
    {
      //---- SOMETIMES THE WIN32 API CALL FAILS FOR NO GOOD REASON.
      //---- TRY FOR THE USUAL 1 MSEC MINIMUM; OTHERWISE,
      //---- MULTITHREADING CAN BE SLOWED SIGNIFICANTLY.
      timeBeginPeriod (1);
      Sleep (35);     //-- 35 milliseconds
    }
#endif

#else     //-- HAVE_REALTIME_CLOCK IS UNDEFINED
    _nNumTimers = 0;
#endif

    return;
  }

//----------------------------------------------------------------------
//  Destroy
//----------------------------------------------------------------------
  void SystemTimer::destroy()
  {
    if (_nNumTimers == 0)
      return;

    delete[] _baIsStarted;
    delete[] _daCumTimes;
    delete[] _naNumCalls;

#if defined(HAVE_REALTIME_CLOCK)
    delete[] _taStartTimes;

#if defined(_WIN32)
    //---- UNSET THE TIMER FOR WINDOWS.
    if (_nWindowsSaveTimerMin > 0)
      timeEndPeriod (_nWindowsSaveTimerMin);
#endif
#endif

    return;
  }


//----------------------------------------------------------------------
//  Method getDateTime
//----------------------------------------------------------------------
  void  SystemTimer::getDateTime (std::string &  sCurrentDateTime)
  {
#if defined(_WIN32)
    SYSTEMTIME  tNow;
    GetLocalTime (&tNow);
    std::ostringstream datetime;
    datetime << std::setw(2)                      << tNow.wMonth  << "/";
    datetime << std::setw(2) << std::setfill('0') << tNow.wDay    << "/";
    datetime << std::setw(4)                      << tNow.wYear   << " ";
    datetime << std::setw(2) << std::setfill('0') << tNow.wHour   << ":";
    datetime << std::setw(2) << std::setfill('0') << tNow.wMinute << ":";
    datetime << std::setw(2) << std::setfill('0') << tNow.wSecond;
    sCurrentDateTime = datetime.str();
#else
    time_t  tNow = time (NULL);
    struct tm  tConvertedNow;
    if (localtime_r (&tNow, &tConvertedNow) == NULL)
    {
      sCurrentDateTime = "Error getting time";
    }
    else
    {
      std::ostringstream datetime;
      datetime << std::setw(2)                      << tConvertedNow.tm_mon + 1     << "/";
      datetime << std::setw(2) << std::setfill('0') << tConvertedNow.tm_mday        << "/";
      datetime << std::setw(4)                      << tConvertedNow.tm_year + 1900 << " ";
      datetime << std::setw(2) << std::setfill('0') << tConvertedNow.tm_hour        << ":";
      datetime << std::setw(2) << std::setfill('0') << tConvertedNow.tm_min         << ":";
      datetime << std::setw(2) << std::setfill('0') << tConvertedNow.tm_sec;
      sCurrentDateTime = datetime.str();
    }
#endif

    return;
  }


//----------------------------------------------------------------------
//  Method sleepMilliSecs
//----------------------------------------------------------------------
  void  SystemTimer::sleepMilliSecs (const int  nMilliSecs)
  {
#if defined(_WIN32)
    //---- NOTE, WINDOWS DOES SOMETHING EVEN IF ZERO.
    Sleep ((DWORD) nMilliSecs);
#else
    struct timespec  cTS;
    cTS.tv_sec  = nMilliSecs / 1000;
    cTS.tv_nsec = (nMilliSecs % 1000) * 1000000;
    nanosleep (&cTS, NULL);
#endif

    return;
  }


//-----------------------------------------------------------------------------
//  Method start
//-----------------------------------------------------------------------------
  bool  SystemTimer::start (const int  nTimerID)
  {
    if ((nTimerID < 0) || (nTimerID >= _nNumTimers))
      return( false );

#if defined(HAVE_REALTIME_CLOCK)
    //---- READ AND STORE THE CURRENT TIME.
#if defined(_WIN32)
    _taStartTimes[nTimerID].nnRealTime = timeGetTime();

#else
    gettimeofday (&(_taStartTimes[nTimerID].cRealTime), NULL);

#endif

    _baIsStarted[nTimerID] = true;
    return( true );
#else
    return( false );
#endif
  }


//-----------------------------------------------------------------------------
//  Method stop
//-----------------------------------------------------------------------------
  bool  SystemTimer::stop (const int  nTimerID)
  {
    if ((nTimerID < 0) || (nTimerID >= _nNumTimers))
      return( false );

    if (_baIsStarted[nTimerID] == false)
      return( false );

    if (_fence)
      Kokkos::fence();

    //---- ADD ELAPSED TIME SINCE THE LAST CALL TO start().
    _daCumTimes[nTimerID] += getTimeSinceLastStart_ (nTimerID);
    _baIsStarted[nTimerID] = false;
    _naNumCalls[nTimerID]++;

    return( true );
  }


//-----------------------------------------------------------------------------
//  Method getTotalTime
//-----------------------------------------------------------------------------
  double  SystemTimer::getTotalTime (const int  nTimerID) const
  {
    if ((nTimerID < 0) || (nTimerID >= _nNumTimers))
      return( -1.0 );

    if ((getNumStarts (nTimerID) == 0) && (_baIsStarted[nTimerID] == false))
      return( 0.0 );

    double  dResult = _daCumTimes[nTimerID];
    if (_baIsStarted[nTimerID] == true)
      dResult += getTimeSinceLastStart_ (nTimerID);

    if (_pmap != nullptr)
      dResult = _pmap->gridAllReduce(dResult, ProcessorMap::Max);

    return( dResult );
  }


//-----------------------------------------------------------------------------
//  Method getNumStarts
//-----------------------------------------------------------------------------
  int  SystemTimer::getNumStarts (const int  nTimerID) const
  {
    if ((nTimerID < 0) || (nTimerID >= _nNumTimers))
      return( -1 );

    return( _naNumCalls[nTimerID] );
  }


//-----------------------------------------------------------------------------
//  Method getAvgTime
//-----------------------------------------------------------------------------
  double  SystemTimer::getAvgTime (const int  nTimerID) const
  {
    if ((nTimerID < 0) || (nTimerID >= _nNumTimers))
      return( -1.0 );

    if (getNumStarts (nTimerID) == 0)
      return( 0.0 );

    double dResult = _daCumTimes[nTimerID] / ((double) getNumStarts (nTimerID));

    if (_pmap != nullptr)
      dResult = _pmap->gridAllReduce(dResult, ProcessorMap::Max);

    return dResult;
  }


//-----------------------------------------------------------------------------
//  Method reset
//-----------------------------------------------------------------------------
  void  SystemTimer::reset (const int  nTimerID)
  {
    if ((nTimerID < 0) || (nTimerID >= _nNumTimers))
      return;

    _daCumTimes[nTimerID] = 0.0;
    _baIsStarted[nTimerID] = false;
    _naNumCalls[nTimerID] = 0;
    return;
  }


//-----------------------------------------------------------------------------
//  Private Method getTimeSinceLastStart_
//-----------------------------------------------------------------------------
  double  SystemTimer::getTimeSinceLastStart_ (const int  nTimerID) const
  {
#if defined(HAVE_REALTIME_CLOCK)
    //---- READ THE CURRENT TIME AND COMPUTE ELAPSED TIME.
#if defined(_WIN32)
    //---- WINDOWS TIMER RETURNS MILLISECOND TICKS.
    DWORD  nnNow = timeGetTime();
    DWORD  nnElapsed = nnNow - _taStartTimes[nTimerID].nnRealTime;
    return( ((double) nnElapsed) / 1000.0 );

#else
    timeval  cNow;
    gettimeofday (&cNow, NULL);
    time_t  nNowSecs = cNow.tv_sec;
    time_t  nStartSecs = _taStartTimes[nTimerID].cRealTime.tv_sec;
    long int  nnNowUsecs = cNow.tv_usec;
    long int  nnStartUsecs = _taStartTimes[nTimerID].cRealTime.tv_usec;

    time_t  nDiffSecs = nNowSecs - nStartSecs;
    long int  nnDiffUsecs;
    if (nnNowUsecs >= nnStartUsecs)
      nnDiffUsecs = nnNowUsecs - nnStartUsecs;
    else
    {
      nDiffSecs--;
      nnDiffUsecs = 1000000 - (nnStartUsecs - nnNowUsecs);
    }
    double  dDiff =   ((double) nDiffSecs)
      + ((double) nnDiffUsecs) * 1.0e-6;
    return( dDiff );

#endif

#else     //-- HAVE_REALTIME_CLOCK IS UNDEFINED
    return( 0.0 );
#endif
  }


}     //-- namespace Genten
