//======================================================================================================
// Copyright 2012, NaturalPoint Inc.
//======================================================================================================
#pragma once

// System includes
#include <string.h> //== for memcpy
#include <math.h>
#include <float.h>
#include <iostream>
#include <algorithm>

// Local includes
#include "Platform.h"
#if !defined(__PLATFORM__LINUX__)

#include "Core/BuildConfig.h"
#include "Core/EulerTypes.h"

namespace Core
{
    template<typename T>
    class cVector3;

    template<typename T, bool AutoNormalize>
    class cQuaternion;

    // 4 x 4 Row Major Matrix
    template<typename T>
    class CORE_API cMatrix4
    {
    public:
        cMatrix4();
        cMatrix4( const cMatrix4& other );
        cMatrix4( T t11, T t12, T t13, T t14, T t21, T t22, T t23, T t24, T t31, T t32, T t33, T t34, T t41, T t42, T t43, T t44 );
        cMatrix4( const T vals[16] );

        /// <summary>Set the vector values.</summary>
        void            SetValues( T t11, T t12, T t13, T t14, T t21, T t22, T t23, T t24, T t31, T t32, T t33, T t34, T t41, T t42, T t43, T t44 );

        /// <summary>Set the vector values.</summary>
        void            SetValues( const T vals[16] );

        /// <summary>Returns the value at row(0 to 3) and col(0 to 3).</summary>
        T               Value( int row, int col ) const; 

        /// <summary>Returns the value at row(0 to 3) and col(0 to 3).</summary>
        T&              Value( int row, int col );

        /// <summary>Access to the data array.</summary>
        const float*    Data() const;

        /// <summary>Translation matrix.</summary>
        void            Translate( T x, T y, T z );

        /// <summary>Translation matrix.</summary>
        void            Translate( const cVector3<T>& v );

        /// <summary>Translation matrix.</summary>
        void            Translate( const cMatrix4& m );

        /// <summary>X axis rotation matrix.</summary>
        void            RotateX( T angle );

        /// <summary>Y axis rotation matrix.</summary>
        void            RotateY( T angle );

        /// <summary>Z axis rotation matrix.</summary>
        void            RotateZ( T angle );

        /// <summary>Rotation matrix with specfied rotation order.</summary>
        void            Rotate( T x, T y, T z, int rotationOrder );

        /// <summary>Rotation matrix from a quaternion.</summary>
        void            Rotate( const cQuaternion<T, false> &q );

        /// <summary>Rotation matrix from another matrix.</summary>
        void            Rotate( const cMatrix4& m );

        /// <summary>Scale matrix.</summary>
        void            Scale( T x, T y, T z );

        /// <summary>Scale matrix.</summary>
        void            Scale( const cVector3<T>& v );

        /// <summary>Combined rotation and translation matrix.</summary>
        void            RotateTranslate( const cQuaternion<T, false>& q, const cVector3<T>& v );

        /// <summary>Combined scale, rotation and translation matrix.</summary>
        void            ScaleRotateTranslate( const cVector3<T>& s, const cQuaternion<T, false>& q, const cVector3<T>& v );

        T               Determinant() const;

        void            Invert();

        void            Invert( const cMatrix4& m );

        void            Transpose();

        void            Transpose( const cMatrix4& m );

        void            Multiply( const cMatrix4& m );

        void            Multiply( const cMatrix4& m1, const cMatrix4& m2 );


        //====================================================================================
        // Type conversion helpers
        //====================================================================================

        cMatrix4<float> ToFloat() const;

        cMatrix4<double> ToDouble() const;

        //====================================================================================
        // Operators
        //====================================================================================

        bool            operator == ( const cMatrix4& rhs ) const;
        bool            operator != ( const cMatrix4& rhs ) const ;
        cMatrix4&       operator *= ( const cMatrix4& rhs );
        cMatrix4        operator * ( const cMatrix4& rhs ) const;

        //====================================================================================
        // Helper constants
        //====================================================================================
        
		static const cMatrix4 kZero;
		static const cMatrix4 kIdentity;

    private:
                    /*  0  1  2  3 */
                    /*  4  5  6  7 */
                    /*  8  9 10 11 */
                    /* 12 13 14 15 */
        T               mVals[16];
    };

    //=========================================================================
    // Stream I/O operators
    //=========================================================================
    template< typename T>
    std::wostream& operator <<( std::wostream& os, const cMatrix4<T>& m )
    {
        os << L"|" << m.Value( 0, 0 ) << L"," << m.Value( 0, 1 ) << L"," << m.Value( 0, 2 ) << L"," << m.Value( 0, 3 ) << L"|\n";
        os << L"|" << m.Value( 1, 0 ) << L"," << m.Value( 1, 1 ) << L"," << m.Value( 1, 2 ) << L"," << m.Value( 1, 3 ) << L"|\n";
        os << L"|" << m.Value( 2, 0 ) << L"," << m.Value( 2, 1 ) << L"," << m.Value( 2, 2 ) << L"," << m.Value( 2, 3 ) << L"|\n";
        os << L"|" << m.Value( 3, 0 ) << L"," << m.Value( 3, 1 ) << L"," << m.Value( 3, 2 ) << L"," << m.Value( 3, 3 ) << L"|\n";
        return os;
    }   

    template< typename T>
    std::ostream& operator <<( std::ostream& os, const cMatrix4<T>& m )
    {
        os << "|" << m.Value( 0, 0 ) << "," << m.Value( 0, 1 ) << "," << m.Value( 0, 2 ) << "," << m.Value( 0, 3 ) << "|\n";
        os << "|" << m.Value( 1, 0 ) << "," << m.Value( 1, 1 ) << "," << m.Value( 1, 2 ) << "," << m.Value( 1, 3 ) << "|\n";
        os << "|" << m.Value( 2, 0 ) << "," << m.Value( 2, 1 ) << "," << m.Value( 2, 2 ) << "," << m.Value( 2, 3 ) << "|\n";
        os << "|" << m.Value( 3, 0 ) << "," << m.Value( 3, 1 ) << "," << m.Value( 3, 2 ) << "," << m.Value( 3, 3 ) << "|\n";
        return os;
    }

    template <typename T>
    const cMatrix4<T> cMatrix4<T>::kZero( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
    template <typename T>
    const cMatrix4<T> cMatrix4<T>::kIdentity( 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 );

    typedef cMatrix4<float> cMatrix4f;
    typedef cMatrix4<double> cMatrix4d;
}

#include "Core/Matrix4.inl"

#endif // __PLATFORM__LINUX__


