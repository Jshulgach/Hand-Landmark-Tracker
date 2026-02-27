//======================================================================================================
// Copyright 2015, NaturalPoint Inc.
//======================================================================================================
#pragma once

// System includes
#include <vector>
#include <map>

// Local includes
#include "Core/CameraRay.h"

namespace Core
{
    /// <summary>
    /// A container class that holds a collection of rays and caches some state information on them
    /// for fast computational access to lists of markers that meet certain criteria.
    /// </summary>
    class cRayBundle
    {
    public:
        typedef std::vector<Core::cCameraRay>::const_iterator RayIterator;
        typedef std::pair<RayIterator,RayIterator> RayIteratorPair;

        cRayBundle() { }
        cRayBundle( const cRayBundle& other );
        ~cRayBundle() { }

        /// <summary>Assignment operator.</summary>
        cRayBundle&     operator=( const cRayBundle& other );

        /// <summary>
        /// Clears all data and prepares this instance to be reused. This is useful for reusing memory
        /// that has been previously allocated.
        /// </summary>
        void            Clear();

        /// <summary>Returns true if there are no rays in the bundle.</summary>
        bool            Empty() const { return mAllRays.empty(); }

        /// <summary>
        /// Populate the rays for this frame instance. The passed array is a non-const reference
        /// because it will be swapped with the internal array and cleared on return, without freeing memory.
        /// That will allow memory to be reused when possible.
        /// </summary>
        void            SetRays( std::vector<Core::cCameraRay>& rays );

        /// <summary>Retrieve begin/end iterators for the full ray list.</summary>
        std::pair<RayIterator,RayIterator> AllRays() const;

        /// <summary>Returns true if there are rays present.</summary>
        bool            HasRays() const { return !( mAllRays.empty() ); }

        /// <summary>Retrieve begin/end iterators for the assigned rays (rays that are assigned to markers).</summary>
        std::pair<RayIterator,RayIterator> AssignedRays() const;

        /// <summary>Returns true if there are assigned rays present.</summary>
        bool            HasAssignedRays() const { return ( ReconstructionRays( 0 ).second != mAllRays.end() ); }

        /// <summary>Retrieve begin/end iterators for the unassigned rays (rays that are NOT assigned to markers).</summary>
        std::pair<RayIterator,RayIterator> UnassignedRays() const;

        /// <summary>Returns true if there are assigned rays present.</summary>
        bool            HasUnassignedRays() const
        {
            RayIteratorPair result = ReconstructionRays( 0 );
            return ( result.first != result.second );
        }

        /// <summary>
        /// Return an iterator pair for the rays that contribute to the given reconstruction.
        /// </summary>
        std::pair<RayIterator,RayIterator> ReconstructionRays( unsigned int reconstructionID ) const;

        /// <summary>Returns true if there are assigned rays present for the given reconstruction.</summary>
        bool            HasReconstructionRays( unsigned int reconstructionID ) const
        {
            RayIteratorPair result = ReconstructionRays( reconstructionID );
            return ( result.first != result.second );
        }

    private:
        // Array of all rays.
        std::vector<Core::cCameraRay> mAllRays;

        // Map from reconstruction ID's to RayIterator pairs
        std::map<unsigned int,std::pair<RayIterator,RayIterator>> mReconstructionRays;

        void            ParseReconstructionAssignments();
    };
}


