using UnityEngine;
using System.Collections;
using System;
using System.Runtime.InteropServices;

public class VRPNPlugin : MonoBehaviour
{
    //Lets make our calls from the Plugin
    [DllImport("OptiTrackPlugin")]
    private static extern void VRPNStartup();
    [DllImport("OptiTrackPlugin")]
    private static extern void VRPNTick();
    [DllImport("OptiTrackPlugin")]
    private static extern float VRPNPositionX();
    [DllImport("OptiTrackPlugin")]
    private static extern float VRPNPositionY();
    [DllImport("OptiTrackPlugin")]
    private static extern float VRPNPositionZ();
    [DllImport("OptiTrackPlugin")]
    private static extern float VRPNOrientX();
    [DllImport("OptiTrackPlugin")]
    private static extern float VRPNOrientY();
    [DllImport("OptiTrackPlugin")]
    private static extern float VRPNOrientZ();
    [DllImport("OptiTrackPlugin")]
    private static extern float VRPNOrientW();

	public Vector3 basePos;
	void Loop()
	{
	}

	void Update()
	{
		VRPNTick();
		VRPNTick();
		VRPNTick();
		VRPNTick();
		Debug.Log(PosX());
		
		var cam = GameObject.Find("MainCamera");
		
		Vector3 pos = cam.transform.position;
		Quaternion ori = cam.transform.rotation;
		
		pos.x = PosX()*18+basePos.x;
		pos.y = PosY()*18+basePos.y;
		pos.z = -PosZ()*18+basePos.z;
		cam.transform.position = pos;
		
		ori.x  = OriX();
		ori.y  = OriY();
		ori.z  = -OriZ();
		ori.w = -OriW();
		cam.transform.rotation = ori;
	}

    void Start()
    {
        VRPNStartup();
		var cam = GameObject.Find("MainCamera");
		basePos = cam.transform.position;
    }
    void Tick()
    {
        VRPNTick();
    }
    float PosX()
    {
        return VRPNPositionX();
    }
    float PosY()
    {
        return VRPNPositionY();
    }
    float PosZ()
    {
        return VRPNPositionZ();
    }

    float OriX()
    {
        return VRPNOrientX();
    }
    float OriY()
    {
        return VRPNOrientY();
    }
    float OriZ()
    {
        return VRPNOrientZ();
    }
    float OriW()
    {
        return VRPNOrientW();
    }
}

