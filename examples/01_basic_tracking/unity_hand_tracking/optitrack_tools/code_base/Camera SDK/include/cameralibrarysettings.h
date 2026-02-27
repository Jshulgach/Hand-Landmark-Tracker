
//====================================================================================================-----
//== Copyright NaturalPoint, All Rights Reserved
//====================================================================================================-----

//== these are top level settings for the Camera SDK.  these settings need to be set as early as possible
//== during program execution and will not have any effect if changes to these settings are made after
//== any calls into the CameraManager class.

#ifndef __CAMERALIBRARY__CAMERALIBRARYSETTINGS_H__
#define __CAMERALIBRARY__CAMERALIBRARYSETTINGS_H__

#include "singleton.h"

class CLAPI cCameraLibraryStartupSettings : public CameraLibrary::Singleton<cCameraLibraryStartupSettings>
{
public:
    cCameraLibraryStartupSettings();

    enum eUSBDevices
    {
        USB_Disabled = 0,
        USB_Licensing_Only,
        USB_Enabled
    };

    void EnableUSBDevices(eUSBDevices EnableState);
    eUSBDevices USBDevices() const;

	void EnableEthernetDevices(bool Enable);
	bool IsEthernetDevicesEnabled() const;

	void EnableDevelopment();
	bool IsDevelopmentEnabled() const;

    void SetEnableFilterSwitch(bool Enable);
    bool IsFilterSwitchEnabled() const;

    enum eLLDPOptions
    {
        LLDP_Automatic = 0,
        LLDP_Override
    };

    void SetLLDPDetectionDefault( eLLDPOptions lldpSetting );
    eLLDPOptions LLDPDetectionDefault() const;

};

#endif
