//======================================================================================================
// Copyright 2015, NaturalPoint Inc.
//======================================================================================================
#pragma once

// System includes
#include <vector>

// Local includes
#include "Core/Marker.h"

namespace Core
{
    /// <summary>
    /// A container class that holds a collection of markers and caches some state information on them
    /// for fast computational access to lists of markers that meet certain criteria.
    /// </summary>
    class cMarkerBundle
    {
    public:
        typedef std::vector<Core::cMarker>::const_iterator MarkerIterator;
        typedef std::pair<MarkerIterator,MarkerIterator> MarkerIteratorPair;

        cMarkerBundle();
        cMarkerBundle( const cMarkerBundle& other );
        ~cMarkerBundle() { }

        /// <summary>Assignment operator.</summary>
        cMarkerBundle&     operator=( const cMarkerBundle& other );

        /// <summary>Clears all data and prepares this instance to be reused. This is useful for reusing memory
        /// that has been previously allocated </summary>
        void            Clear();

        /// <summary>Returns true if there are no rays in the bundle.</summary>
        bool            Empty() const { return mAllMarkers.empty(); }

        /// <summary>
        /// Populate the rays for this frame instance. The passed array is a non-const reference
        /// because it will be swapped with the internal array and cleared on return, without freeing memory.
        /// That will allow memory to be reused when possible.
        /// The selection order array gives the ID's of selected markers in the order in which they were selected.
        /// </summary>
        void            SetMarkers( std::vector<Core::cMarker>& markers, const std::vector<Core::cUID>& selectionOrder );

        /// <summary>Retrieve begin/end iterators for the full ray list.</summary>
        MarkerIteratorPair AllMarkers() const;

        /// <summary>Retrieve total number of all markers.</summary>
        size_t          MarkerCount() const;

        /// <summary>Attempt to find a marker with the given ID. Returns false if not found.</summary>
        bool            FindMarker( const Core::cUID& id, Core::cMarker* oMarker ) const;

        /// <summary>Retrieve begin/end iterators for all selected markers.</summary>
        MarkerIteratorPair SelectedMarkers() const;

        /// <summary>Returns true if there are any selected markers present.</summary>
        bool            HasSelectedMarkers() const { return ( !mSelectedMarkers.empty() ); }

        /// <summary>Retrieve total number of all markers.</summary>
        size_t          SelectedMarkerCount() const;

        /// <summary>Retrieve the primary selected marker, or null if none.</summary>
        const Core::cMarker* PrimarySelectedMarker() const;

        /// <summary>Retrieve begin/end iterators for all labeled markers.</summary>
        MarkerIteratorPair LabeledMarkers() const;

        /// <summary>Returns true if there are any unlabeled markers present.</summary>
        bool            HasLabeledMarkers() const { return ( mFirstLabeledMarker != mMarkersByLabeled.end() ); }

        /// <summary>Retrieve begin/end iterators for all unlabeled markers.</summary>
        MarkerIteratorPair UnlabeledMarkers() const;

        /// <summary>Returns true if there are any labeled markers present.</summary>
        bool            HasUnlabeledMarkers() const { return ( mMarkersByLabeled.begin() != mFirstLabeledMarker ); }

    private:
        // Array of markers
        std::vector<Core::cMarker> mAllMarkers;

        // Array of selected markers, in correct selection order.
        std::vector<Core::cMarker> mSelectedMarkers;

        // Arrays of labeled/unlabeled marker pointers.
        std::vector<Core::cMarker> mMarkersByLabeled;

        // Iterator pointing to the first labeled marker, which is also the end of the unlabeled marker list.
        MarkerIterator  mFirstLabeledMarker;

        // TODO : KNN Tree of markers
    };
}

