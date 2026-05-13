//======================================================================================================
// Copyright 2012, NaturalPoint Inc.
//======================================================================================================
#pragma once

#include <string>

// Local includes
#include "BuildConfig.h"

namespace Core
{
    struct CORE_API sTimeCode
    {
        sTimeCode() : TimeCode( 0 ), TimeCodeSubFrame( 0 ), TimeCodeDropFrame( false ), Valid( false ) { }

        unsigned int    TimeCode;
        unsigned int    TimeCodeSubFrame;
        bool            TimeCodeDropFrame;
        bool            Valid;

        int             Hours() const;
        int             Minutes() const;
        int             Seconds() const;
        int             Frame() const;
        int             SubFrame() const;
        bool            IsDropFrame() const;

        void            Stringify( char *buffer, int bufferSize ) const;
        std::string     Stringify() const;
    };
}


