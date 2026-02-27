//======================================================================================================
// Copyright 2015, NaturalPoint Inc.
//======================================================================================================
#pragma once

// System includes
#include <vector>

// Local includes
#include "Core/RigidBody.h"

namespace Core
{
    /// <summary>
    /// A container class that holds a collection of rigid body solutions.
    /// </summary>
    class cRigidBodyBundle
    {
    public:
        typedef std::vector<Core::cRigidBody>::const_iterator RigidBodyIterator;
        typedef std::pair<RigidBodyIterator,RigidBodyIterator> RigidBodyIteratorPair;

        cRigidBodyBundle();
        cRigidBodyBundle( const cRigidBodyBundle& other );
        ~cRigidBodyBundle() { }

        /// <summary>Assignment operator.</summary>
        cRigidBodyBundle& operator=( const cRigidBodyBundle& other );

        /// <summary>Clears all data and prepares this instance to be reused. This is useful for reusing memory
        /// that has been previously allocated </summary>
        void            Clear();

        /// <summary>Returns true if there are no rigid body solutions.</summary>
        bool            Empty() const { return mAllRigidBodies.empty(); }

        /// <summary>
        /// Populate the rigid body solutions for this frame instance. The passed array is a non-const reference
        /// because it will be swapped with the internal array and cleared on return, without freeing memory.
        /// That will allow memory to be reused when possible.
        /// </summary>
        void            SetRigidBodies( std::vector<Core::cRigidBody>& rigidBodies );

        /// <summary>Retrieve begin/end iterators for the full rigid body solution list.</summary>
        RigidBodyIteratorPair AllRigidBodies() const;

        /// <summary>Retrieve the total number of rigid bodies.</summary>
        size_t          RigidBodyCount() const { return mAllRigidBodies.size(); }

        /// <summary>Retrieve begin/end iterators for all the tracked rigid bodies.</summary>
        RigidBodyIteratorPair TrackedRigidBodies() const;

        /// <summary>Retrieve begin/end iterators for all the untracked rigid bodies.</summary>
        RigidBodyIteratorPair UntrackedRigidBodies() const;

        /// <summary>Retrieve begin/end iterators for all the selected rigid bodies.</summary>
        RigidBodyIteratorPair SelectedRigidBodies() const;

        /// <summary>Retrieve the primary selected rigid body, or null if none.</summary>
        const Core::cRigidBody* PrimarySelectedRigidBody() const;

        /// <summary>Retrieve begin/end iterators for all the unselected rigid bodies.</summary>
        RigidBodyIteratorPair UnselectedRigidBodies() const;

        /// <summary>Attempt to find a rigid body with the given ID. Returns false if not found.</summary>
        bool            FindRigidBody( const Core::cUID& id, Core::cRigidBody* oRigidBody ) const;

    private:
        // Array of all rigid bodies, sorted by tracked/untracked.
        std::vector<Core::cRigidBody> mAllRigidBodies;
        int             mSetCount;

        // Copy of rigid bodies arranged by unselected/selected.
        std::vector<Core::cRigidBody> mRigidBodiesByTracked;
        std::vector<Core::cRigidBody> mRigidBodiesBySelection;

        RigidBodyIterator mFirstTrackedRigidBody;
        RigidBodyIterator mFirstSelectedRigidBody;
    };
}

