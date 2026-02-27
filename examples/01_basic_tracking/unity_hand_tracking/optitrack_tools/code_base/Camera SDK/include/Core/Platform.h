//======================================================================================================
// Copyright 2013, NaturalPoint Inc.
//======================================================================================================
#pragma once

#if defined(__unix__)
    #if !defined(__PLATFORM__LINUX__)
        #define __PLATFORM__LINUX__
    #endif
#endif

#ifdef __PLATFORM__LINUX__
#define sprintf_s snprintf
#define byte unsigned char
#endif


namespace Core
{
	///<summary>Query the number of cores available on this machine.</summary>
	int                 CoreCount();
}



