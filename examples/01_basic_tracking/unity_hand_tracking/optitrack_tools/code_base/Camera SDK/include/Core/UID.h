//======================================================================================================
// Copyright 2012, NaturalPoint Inc.
//======================================================================================================
#pragma once

#include <ostream>
#include <iomanip>

#include "Core/BuildConfig.h"

// This is here to allow inclusion of this definition into the Motive API directly, rather than including this header.
// For now, anytime changes are made to this class, this class should also be copied into the Motive API header.
#ifndef _CORE_UID_CLASS
#define _CORE_UID_CLASS

//====================================================================================================================================
//== WARNING == PLEASE READ ORIGINAL COMMENT ABOVE ^^^^^^ === YOU CAN"T MAKE CHANGES TO THIS CLASS WITHOUT CUT & PASTING THEM INTO ===
//== NPTRACKINGTOOLS.H OR VERY BAD THINGS WILL HAPPEN ======== (ESPECIALLY ADDING/REMOVING MEMBER VARIABLES ====================== ===
//====================================================================================================================================

namespace Core
{
    /// <summary>
    /// A platform-neutral 128-bit universal identifier. It is essentially guaranteed to never
    /// generate the same ID twice.
    /// </summary>
    class CORE_API cUID
    {
    public:
        typedef unsigned long long int uint64;

        /// <summary>
        /// Create a default UID. In order to create a UID that has a valid unique identifier you
        /// must call Generate().
        /// </summary>
        cUID() : mHighBits( 0 ), mLowBits( 0 ) { }

        cUID( uint64 high, uint64 low ) : mHighBits( high ), mLowBits( low ) { }
        cUID( const cUID & obj ) : mHighBits( obj.mHighBits ), mLowBits( obj.mLowBits ) { }
        cUID&           operator=( const cUID & rhs ) { mHighBits = rhs.mHighBits; mLowBits = rhs.mLowBits; return *this; }

        /// <summary>
        /// Set the value of the UID from two long integer values. It is up to the caller to ensure that
        /// the resulting UID is unique.
        /// </summary>
        void            SetValue( uint64 highBits, uint64 lowBits ) { mHighBits = highBits; mLowBits = lowBits; }

        /// <summary>Get the low 64 bits of the UID.</summary>
        uint64          LowBits() const { return mLowBits; }

        /// <summary>Get the high 64 bits of the UID.</summary>
        uint64          HighBits() const { return mHighBits; }

        /// <summary>Returns true if the ID is valid.</summary>
        bool            Valid() const { return ( mHighBits != 0 && mLowBits != 0 ); }

        /// <summary>Generate a new UID value.</summary>
        static cUID     Generate();

        //==============================================================================================
        // Comparison operators
        //==============================================================================================

        bool            operator<( const cUID & rhs ) const
        {
            return ( ( mHighBits < rhs.mHighBits ) ? true : ( mHighBits == rhs.mHighBits ? ( mLowBits < rhs.mLowBits ) : false ) );
        }
        bool            operator<=( const cUID & rhs ) const
        {
            return ( ( mHighBits < rhs.mHighBits ) ? true : ( mHighBits == rhs.mHighBits ? ( mLowBits <= rhs.mLowBits ) : false ) );
        }
        bool            operator>( const cUID & rhs ) const { return !( *this <= rhs ); }
        bool            operator>=( const cUID & rhs ) const { return !( *this < rhs ); }
        bool            operator==( const cUID & rhs ) const
        {
            return ( ( mHighBits == rhs.mHighBits ) && ( mLowBits == rhs.mLowBits ) );
        }
        bool            operator!=( const cUID & rhs ) const { return !( *this == rhs ); }

        //==============================================================================================
        // Constants
        //==============================================================================================

        static const cUID Invalid;

    private:
        uint64          mHighBits;
        uint64          mLowBits;
    };
}

//======================================================================================================
// Operators
//======================================================================================================
std::ostream& operator<<( std::ostream& os, const Core::cUID& id );
std::istream& operator>>( std::istream& is, Core::cUID& id );

std::wostream& operator<<( std::wostream& os, const Core::cUID& id );
std::wistream& operator>>( std::wistream& is, Core::cUID& id );


//====================================================================================================================================
//== WARNING == PLEASE READ ORIGINAL COMMENT ABOVE ^^^^^^ === YOU CAN"T MAKE CHANGES TO THIS CLASS WITHOUT CUT & PASTING THEM INTO ===
//== NPTRACKINGTOOLS.H OR VERY BAD THINGS WILL HAPPEN ======== (ESPECIALLY ADDING/REMOVING MEMBER VARIABLES ====================== ===
//====================================================================================================================================

#endif // _CORE_UID_CLASS

