//======================================================================================================
// Copyright 2014, NaturalPoint Inc.
//======================================================================================================
#pragma once

// Local includes
#include "Core/Vector2.h"
#include "Core/Vector3.h"

namespace Core
{
    /// <summary>
    /// A class representing a ray that emanates from a camera, and may be assigned to a 3D marker reconstruction.
    /// </summary>
    class cCameraRay
    {
    public:
        cCameraRay( int id, int cameraID, unsigned int reconstructionID, const Core::cVector2f& imagePosition,
            float imageArea, const Core::cVector3f& origin, const Core::cVector3f& direction, float length ) :
            mID( id ), mCameraID( cameraID ), mReconstructionID( reconstructionID ),
            mImagePosition( imagePosition ), mImageArea( imageArea ),
            mOrigin( origin ), mDirection( direction ), mLength( length ) { }
        virtual ~cCameraRay() { }

        /// <summary>The (unique among rays) ID for this ray.</summary>
        int             ID() const { return mID; }

        /// <summary>Retrieve the ID of the camera from which this ray originated.</summary>
        // TODO: Camera ID may be wrong when merging points from multiple point cloud instances
        int             CameraID() const { return mCameraID; }

        /// <summary>Set the assigned reconstruction ID.</summary>
        void            SetReconstructionID( unsigned int id ) { mReconstructionID = id; }

        /// <summary>The ID of the marker that this ray is assigned to, or cUID::Invalid if none.</summary>
        unsigned int    ReconstructionID() const { return mReconstructionID; }

        /// <summary>Retrieve the 2D image position in the camera image.</summary>
        const Core::cVector2f& ImagePosition() const { return mImagePosition; }

        /// <summary>Retrieves the (pixel) base area in the camera image.</summary>
        float           ImageArea() const { return mImageArea; }

        /// <summary>3D ray origin.</summary>
        const Core::cVector3f& Origin() const { return mOrigin; }

        /// <summary>3D ray direction.</summary>
        const Core::cVector3f& Direction() const { return mDirection; }

        /// <summary>Set the ray length.</summary>
        void            SetLength( float len ) { mLength = len; }

        /// <summary>Ray length.</summary>
        float           Length() const { return mLength; }

        /// <summary>Returns true if this ray instance is identical to the given instance.</summary>
        bool            Equals( const cCameraRay& ray ) const { return mID == ray.mID && mCameraID == ray.mCameraID; }

        /// <summary>Comparison operator. Used mostly for sorting.</summary>
        bool            operator<( const cCameraRay& other ) const { return ( mReconstructionID < other.mReconstructionID ); }
        bool            operator<( unsigned int id ) const { return ( mReconstructionID < id ); }
        friend bool     operator<( unsigned int id, const cCameraRay& other ) { return ( id < other.mReconstructionID ); }

    private:
        // An ID that is unique within the context of camera rays.
        int             mID;

        // The ID (serial number) of the camera that originated this ray.
        int             mCameraID;

        // The ID of the reconstruction this ray is assigned to, or 0 if the ray is unassigned.
        unsigned int    mReconstructionID;

        // The (pixel) area of the ray origin in the camera image.
        float           mImageArea;

        // The 2D coordinates of the ray origin in the camera image.
        Core::cVector2f mImagePosition;

        // The 3D ray origin.
        Core::cVector3f mOrigin;

        // The normalized 3D ray direction
        Core::cVector3f mDirection;

        // The length of the ray.
        float           mLength;
    };
}


