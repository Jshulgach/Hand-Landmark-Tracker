//======================================================================================================
// Copyright 2012, NaturalPoint Inc.
//======================================================================================================
#pragma once

// System includes
#include <string>
#include <vector>

// Local includes
#include "Core/UID.h"

namespace Core
{
    class cLabel
    {
    public:
        enum eEntityType
        {
            None = 0,      // Means an entity that is NOT labeled
            MarkerSet,     // Labeled marker set markers
            Skeleton,      // Skeleton or skeleton markers
            RigidBody,     // Rigid bodies or rigid body markers
            Joint          // Skeleton joints
        };

        cLabel();
        cLabel( const std::wstring &entityName, const std::wstring &memberName, eEntityType type, unsigned int entityID,
            unsigned int memberIdx );
        cLabel( const std::wstring &entityName, const std::wstring &memberName, const Core::cUID &id );
        cLabel( const cLabel &other );

        cLabel&         operator=( const cLabel &other );

        /// <summary>Set the name associated with the label.</summary>
        void            SetName( const std::wstring &name );

		/// <summary>Set the names associated with the label.</summary>
		void            SetNames( const std::wstring &entityName, const std::wstring &memberName );

        /// <summary>Human-readable string identifying the entity this label applies to.</summary>
        const std::wstring& EntityName() const { return mEntityName; }

		/// <summary>Set member name.</summary>
		void            SetMemberName( const std::wstring &memberName );
        
		/// <summary>Human-readable string identifying the member of the entity that this label applies to.</summary>
        const std::wstring& MemberName() const { return mMemberName; }

        /// <summary>The fully qualified name to use to identify this label.</summary>
        std::wstring    Name() const;

        /// <summary>Parse fully qualified name into entity and member names.</summary>
        static void     ParseName( const std::wstring &name, std::wstring &entityName, std::wstring &memberName );

        /// <summary>Entity type.</summary>
        eEntityType     Type() const { return mType; }

        /// <summary>Which entity in the list of available entities that this label applies to.</summary>
        unsigned int    EntityID() const { return mEntityID; }

        /// <summary>The index of the element within the entity that this label applies to.</summary>
        unsigned int    MemberID() const { return mMemberID; }

        /// <summary>Returns the encoded ID based on the various contained types and indices.</summary>
        Core::cUID      EncodedUID() const { return mEncodedUID; }

        /// <summary>Returns true if the passed cUID is a valid label.</summary>
        static bool     IsLabel( const Core::cUID &uid, bool checkForValidType = false );

        /// <summary>Decodes a cUID into component parts. If the given cUID is not a label, returns false.</summary>
        /// <param name="uid">Input cUID to be decoded.</param>
        /// <param name="type">Output entity type decoded from cUID.</param>
        /// <param name="entityID">Output entity ID decoded from cUID.</param>
        /// <param name="memberID">Output member ID decoded from cUID.</param>
        static bool     DecodeUID( const Core::cUID &uid, eEntityType &type, unsigned int &entityID, unsigned int &memberID );
        static bool     DecodeUID( const Core::cUID &uid, unsigned int &entityID, unsigned int &memberID );

        /// <summary>Convert to a legacy ID (int) from a cUID.</summary>
        static int      ToLegacyID( const Core::cUID &uid );

        /// <summary>Convert a legacy ID to a cUID, given an entity type.</summary>
        static Core::cUID FromLegacyID( eEntityType entType, int id );

        /// <summary>Convert a legacy ID to a cUID, given an entity type and member index.</summary>
        static Core::cUID EncodeUID( eEntityType entType, unsigned int entityID, unsigned int memberIdx );
       
        // Convenience constants
        static const cLabel Invalid;

        // cLabel identifier
        static const long long klabelIdentifier;
        static const long long kTypeMask;

    private:
        std::wstring    mEntityName;
        std::wstring    mMemberName;
        eEntityType     mType;
        unsigned int    mEntityID;
        unsigned int    mMemberID;

        // Cached encoded cUID
        Core::cUID      mEncodedUID;
    };
}


