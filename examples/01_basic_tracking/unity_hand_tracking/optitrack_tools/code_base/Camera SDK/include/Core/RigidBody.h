//======================================================================================================
// Copyright 2014, NaturalPoint Inc.
//======================================================================================================
#pragma once

// Local includes
#include "Core/UID.h"
#include "Core/Vector3.h"
#include "Core/Quaternion.h"
#include "Core/Marker.h"

namespace Core
{
    class cRigidBody
    {
    public:
        static const int kMaxRigidBodyMarkers = 20;

        cRigidBody();

        cUID            ID;             //== Marker ID/Label
        bool            Selected;       //== Selection state
        int             MarkerCount;

        //== rigid body tracked / untracked information ==--

        bool            Tracked;
        int             FramesUntracked;

        //== position and orientation ==--

        void            SetPosition( const cVector3f& pos ) { mPosition = pos; }
        const cVector3f& Position() const { return mPosition; }

        void            SetRotation( const cQuaternionf& rot ) { mOrientation = rot; }
        const cQuaternionf& Rotation() const { return mOrientation; }

        //== individual marker information ==--

        Core::cMarker   Markers[kMaxRigidBodyMarkers];      // World-space calculated marker locations
        bool            MarkerTracked[kMaxRigidBodyMarkers];
        float           MarkerQuality[kMaxRigidBodyMarkers];

        //== additional information ==--

        float           ErrorPerMarker;

        // To be deprecated.
        // TODO : Move this functionality to the LocalTransform class.
        Core::cVector3f WorldToRigidBody( const cVector3f& wsPos ) const;

    private:
        cVector3f       mPosition;
        cQuaternionf    mOrientation;
    };
}


