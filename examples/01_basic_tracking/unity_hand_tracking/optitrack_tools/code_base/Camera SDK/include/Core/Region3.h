//======================================================================================================
// Copyright 2016, NaturalPoint Inc.
//======================================================================================================
#pragma once

// Local includes
#include "Core/Vector3.h"

namespace Core
{
    /// <summary>A cubic 3D region.</summary>

    template<typename T>
    class CORE_API cRegion3
    {
    public:
        /// <summary>A 3D region.</summary>
        cRegion3() : mLowerBound( 0, 0, 0 ) , mUpperBound( 0, 0, 0 ) { }

        /// <summary>A 3D region bound over range x = { minX->maxX }, y = { minY, maxY }, z = { minZ, maxZ }.</summary>
        cRegion3( T minX, T minY, T minZ, T maxX, T maxY, T maxZ ) : mLowerBound( minX, minY, minZ ),mUpperBound( maxX, maxY, maxZ ) { }

        /// <summary>A 3D region bound over range x = { min.X()->max.X() }, y = { min.Y(), max.Y() }, z = { min.Z(), max.Z() }.</summary>
        cRegion3( cVector3<T> min,cVector3<T> max ) : mLowerBound( min ), mUpperBound( max ) { }

        /// <summary>A 3D region of zero size at location.</summary>
        cRegion3( cVector3<T> location ) : mLowerBound( location ), mUpperBound( location ) { }

        /// <summary>Set the lower bound of the 3D region.</summary>
        void            SetLowerBound( T minX, T minY, T minZ ) { mLowerBound.SetValues( minX, minY, minZ ); }

        /// <summary>Set the upper bound of the 3D region.</summary>
        void            SetUpperBound( T maxX, T maxY, T maxZ ) { mUpperBound.SetValues( maxX, maxY, maxZ ); }

        /// <summary>Set the lower bound of the 3D region.</summary>
        void            SetLowerBound( cVector3<T> lowerBound ) { mLowerBound = lowerBound; }

        /// <summary>Set the upper bound of the 3D region.</summary>
        void            SetUpperBound( cVector3<T> upperBound ) { mUpperBound = upperBound; }

        /// <summary>Fetch the lower bound.</summary>
        const cVector3<T> & LowerBound() const { return mLowerBound; }

        /// <summary>Fetch the upper bound.</summary>
        const cVector3<T> & UpperBound() const { return mUpperBound; }

        /// <summary>Expand region by amount.</summary>
        void            ExpandRegion( T amount )
        {
            if( amount<0 )
            {
                ContractRegion( -amount );
                return;
            }

            mLowerBound = mLowerBound - Core::cVector3<T>( amount, amount, amount );
            mUpperBound = mUpperBound + Core::cVector3<T>( amount, amount, amount );
        }

        /// <summary>Contract region by amount.</summary>
        void            ContractRegion( T amount )
        {
            if( amount<0 )
            {
                ExpandRegion( -amount );
                return;
            }

            if( mLowerBound.X()+amount>mUpperBound.X()-amount )
            {
                mLowerBound.X() = mUpperBound.X() = ( mLowerBound.X()+mUpperBound.X() )/2;
            }
            else
            {
                mLowerBound.X() = mLowerBound.X()+amount;
                mUpperBound.X() = mUpperBound.X()+amount;
            }

            if( mLowerBound.Y()+amount>mUpperBound.Y()-amount )
            {
                mLowerBound.Y() = mUpperBound.Y() = ( mLowerBound.Y()+mUpperBound.Y() )/2;
            }
            else
            {
                mLowerBound.Y() = mLowerBound.Y()+amount;
                mUpperBound.Y() = mUpperBound.Y()+amount;
            }

            if( mLowerBound.Z()+amount>mUpperBound.Z()-amount )
            {
                mLowerBound.Z() = mUpperBound.Z() = ( mLowerBound.Z()+mUpperBound.Z() )/2;
            }
            else
            {
                mLowerBound.Z() = mLowerBound.Z()+amount;
                mUpperBound.Z() = mUpperBound.Z()+amount;
            }
        }

        /// <summary>Expand region to include location x,y,z.</summary>
        void            ExpandRegion( T x, T y, T z )
        {
            if( mLowerBound.X()>x )
            {
                mLowerBound.X() = x;
            }

            if( mLowerBound.Y()>y )
            {
                mLowerBound.Y() = y;
            }

            if( mLowerBound.Z()>z )
            {
                mLowerBound.Z() = z;
            }

            if( mUpperBound.X()<x )
            {
                mUpperBound.X() = x;
            }

            if( mUpperBound.Y()<y )
            {
                mUpperBound.Y() = y;
            }

            if( mUpperBound.Z()<z )
            {
                mUpperBound.Z() = z;
            }
        }

        /// <summary>Expand region to include location.</summary>
        void            ExpandRegion( cVector3<T> location )
        {
            ExpandRegion( location.X(), location.Y(), location.Z() );
        }

        /// <summary>Is location within region.</summary>
        bool            IsWithin( T x, T y, T z ) const
        {
            if( mLowerBound.X()<=x && mLowerBound.Y()<=y && mLowerBound.Z()<=z && mUpperBound.X()>=x && mUpperBound.Y()>=y && mUpperBound.Z()>=z )
            {
                return true;
            }

            return false;
        }

        /// <summary>Is location within region.</summary>
        bool            IsWithin( cVector3<T> location ) const
        {
            return IsWithin( location.X(), location.Y(), location.Z() );
        }

        static const cRegion3 kZero;

    private:
        cVector3<T>  mLowerBound;
        cVector3<T>  mUpperBound;
    };

    template <typename T>
    const cRegion3<T> cRegion3<T>::kZero(0,0);

    typedef cRegion3<int> cRegion3i;
    typedef cRegion3<float> cRegion3f;
    typedef cRegion3<double> cRegion3d;

    /// <summary>A spherical 3D region.</summary>

    template<typename T>
    class CORE_API cSphericalRegion3
    {
    public:
        /// <summary>A spherical 3D region.</summary>
        cSphericalRegion3() : mCenter( 0, 0, 0 ), mRadius( 0 ) { }

        /// <summary>A spherical 3D region centered around { center }.</summary>
        cSphericalRegion3( cVector3<T> center ) : mCenter( center ), mRadius( 0 ) { }

        /// <summary>A spherical 3D region centered around { center } and including up to distance radius.</summary>
        cSphericalRegion3( cVector3<T> center, T radius ) : mCenter( center ), mRadius( radius ) { }

        /// <summary>A spherical 3D region centered around { centerX, centerY }.</summary>
        cSphericalRegion3( T centerX, T centerY, T centerZ ) : mCenter( centerX, centerY, centerZ ), mRadius( 0 ) { }

        /// <summary>A spherical 3D region centered around { centerX, centerY, centerZ } and including up to distance radius.</summary>
        cSphericalRegion3( T centerX, T centerY, T centerZ, T radius ) : mCenter( centerX, centerY, centerZ ), mRadius( radius ) { }

        /// <summary>Set the spherical region.</summary>
        void            SetRegion( T centerX, T centerY, T centerZ, T radius ) { mCenter.SetValues( centerX, centerY, centerZ ); mRadius = radius; }

        /// <summary>Set the spherical region.</summary>
        void            SetRegion( cVector3<T> center, T radius ) { mCenter = center; mRadius = radius; }

        /// <summary>Set the center of the spherical 3D region.</summary>
        void            SetCenter( T centerX, T centerY, T centerZ ) { mCenter.SetValues( centerX, centerY, centerZ ); }

        /// <summary>Set the center of the spherical 3D region.</summary>
        void            SetCenter( cVector3<T> center ) { mCenter = center; }

        /// <summary>Set the radius of the spherical 3D region.</summary>
        void            SetRadius( T radius ) { mRadius = radius; }

        /// <summary>Fetch the center of the spherical 3D region.</summary>
        const cVector3<T> & Center() const { return mCenter; }

        /// <summary>Fetch the radius of the spherical 3D region.</summary>
        const T &       Radius() const { return mRadius; }


        /// <summary>Expand region to include location x,y,z.</summary>
        void            ExpandRegion( T x, T y, T z )
        {
            T distance = Distance( x, y, z );

            if( distance>mRadius )
            {
                mRadius = distance;
            }
        }

        /// <summary>Expand region to include location.</summary>
        void            ExpandRegion( cVector3<T> location )
        {
            ExpandRegion( location.X(), location.Y(), location.Z() );
        }

        /// <summary>Expand region by amount.</summary>
        void            ExpandRegion( T amount )
        {
            if( amount<0 )
            {
                ContractRegion( -amount );
                return;
            }

            mRadius = mRadius + amount;
        }

        /// <summary>Contract region by amount.</summary>
        void            ContractRegion( T amount )
        {
            if( amount<0 )
            {
                ExpandRegion( -amount );
                return;
            }

            mRadius = mRadius - amount;

            if( mRadius<0 )
            {
                mRadius = 0;
            }
        }

        /// <summary>Distance from location x,y,z to center of spherical region.</summary>
        T               Distance( T x, T y, T z ) const
        {
            T squared = ( ( ( x-mCenter.X() )*( x-mCenter.X() ) ) + ( ( y-mCenter.Y() )*( y-mCenter.Y() ) ) + ( ( z-mCenter.Z() )*( z-mCenter.Z() ) ) );

            return sqrt( squared );
        }

        /// <summary>Distance from location  to center of spherical region.</summary>
        T               Distance( cVector3<T> location ) const
        {
            return Distance( location.X(), location.Y(), location.Z() );
        }

        /// <summary>Is location within region.</summary>
        bool            IsWithin( T x, T y, T z ) const
        {
            if( Distance( x, y, z )<mRadius )
            {
                return true;
            }

            return false;
        }

        /// <summary>Is location within region.</summary>
        bool            IsWithin( cVector3<T> location ) const
        {
            return IsWithin( location.X(), location.Y(), location.Z() );
        }

        static const cSphericalRegion3 kZero;

    private:
        cVector3<T>  mCenter;
        T            mRadius;
    };

    template <typename T>
    const cSphericalRegion3<T> cSphericalRegion3<T>::kZero( 0, 0, 0, 0 );

    typedef cSphericalRegion3<int> cSphericalRegion3i;
    typedef cSphericalRegion3<float> cSphericalRegion3f;
    typedef cSphericalRegion3<double> cSphericalRegion3d;
}



