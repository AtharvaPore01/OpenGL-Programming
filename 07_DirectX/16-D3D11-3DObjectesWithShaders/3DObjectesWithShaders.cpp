//header files
#include <Windows.h>
#include <stdio.h>
#include <d3d11.h>
#include <d3dcompiler.h>

#pragma warning(disable : 4838)
#include "XNAMath/xnamath.h"
#include "WICTextureLoader.h"

//libraries
#pragma comment (lib, "user32.lib")
#pragma comment (lib, "gdi32.lib")
#pragma comment (lib, "kernel32.lib")
#pragma comment (lib, "d3d11.lib")
#pragma comment (lib, "D3dcompiler.lib")
#pragma comment (lib, "DirectXTK.lib")

#define WIN_WIDTH	800
#define WIN_HEIGHT	600
#define X (GetSystemMetrics(SM_CXSCREEN) - WIN_WIDTH)/2
#define Y (GetSystemMetrics(SM_CYSCREEN) - WIN_HEIGHT)/2

//global function declaration
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

//global variable declaration
FILE *gpFile_ap = NULL;
char gszLogFileName_ap[] = "Log.txt";

HWND ghwnd;

DWORD dwStyle;
WINDOWPLACEMENT wpPrev = { sizeof(WINDOWPLACEMENT) };

bool gbActiveWindow_ap = false;
bool gbEscapeKeyIsPressed_ap = false;
bool gbFullScreen_ap = false;

float gClearColor[4];
IDXGISwapChain *gpIDXGISwapChain = NULL;
ID3D11Device *gpID3D11Device = NULL;
ID3D11DeviceContext *gpID3D11DeviceContext = NULL;
ID3D11RenderTargetView *gpID3D11RenderTargetView = NULL;
ID3D11DepthStencilView *gpID3D11DepthStencilView = NULL;

ID3D11VertexShader *gpID3D11VertexShader = NULL;
ID3D11PixelShader *gpID3D11PixelShader = NULL;

ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Cube_Position = NULL;
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Cube_Texture = NULL;
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Pyramid_Position = NULL;
ID3D11Buffer *gpID3D11Buffer_VertexBuffer_Pyramid_Texture = NULL;

ID3D11Buffer *gpID3D11Buffer_ConstantBuffer = NULL;
ID3D11InputLayout *gpID3D11InputLayout = NULL;

ID3D11RasterizerState *gpID3D11RasterizerState = NULL;

ID3D11ShaderResourceView *gpID3D11ShaderResourceView_Pyramid_Texture = NULL;
ID3D11SamplerState *gpID3D11SamplerState_Pyramid_Texture = NULL;
ID3D11ShaderResourceView *gpID3D11ShaderResourceView_Cube_Texture = NULL;
ID3D11SamplerState *gpID3D11SamplerState_Cube_Texture = NULL;


struct CBUFFER
{
	XMMATRIX WorldViewProjectionMatrix;
};

XMMATRIX gPerspectiveProjectionMatrix;

float RotationAngle = 0.0f;

//WinMain
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	//Variable declaration
	WNDCLASSEX wndclass_ap;
	HWND hwnd;
	MSG msg;
	TCHAR szAppName_ap[] = TEXT("Direct3D11");
	bool bDone_ap = false;
	int iRet_ap = 0;

	//function declaration
	HRESULT d3dInitialise(void);
	void d3dUninitialise(void);
	void d3dDisplay(void);
	void d3dUpdate(void);
	//code
	//create log file
	if (fopen_s(&gpFile_ap, gszLogFileName_ap, "w") != 0)
	{
		MessageBox(NULL, TEXT("Log File Cannot Be Created\nExitting..."), TEXT("Error"), MB_OK | MB_TOPMOST | MB_ICONSTOP);
		exit(0);
	}
	else
	{
		fprintf_s(gpFile_ap, "Log File is Created Successfully\n");
		fclose(gpFile_ap);
	}
	wndclass_ap.cbSize = sizeof(WNDCLASSEX);
	wndclass_ap.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wndclass_ap.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass_ap.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndclass_ap.hInstance = hInstance;
	wndclass_ap.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass_ap.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass_ap.cbClsExtra = 0;
	wndclass_ap.cbWndExtra = 0;
	wndclass_ap.lpfnWndProc = WndProc;
	wndclass_ap.lpszClassName = szAppName_ap;
	wndclass_ap.lpszMenuName = NULL;

	//Register Class
	RegisterClassEx(&wndclass_ap);

	//CreateWindow
	hwnd = CreateWindowEx(WS_EX_APPWINDOW,
		szAppName_ap,
		TEXT("Direct3D11 : Two 3D Textured Shapes Animating"),
		WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
		X,
		Y,
		WIN_WIDTH,
		WIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);

	ghwnd = hwnd;

	ShowWindow(hwnd, iCmdShow);
	SetForegroundWindow(hwnd);
	SetFocus(hwnd);

	//initialise D3D
	HRESULT hr_ap;
	hr_ap = d3dInitialise();
	if (FAILED(hr_ap))
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "d3dInitialize() is failed. Exitting Now...\n");
		fclose(gpFile_ap);
		DestroyWindow(hwnd);
		hwnd = NULL;
	}
	else
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "d3dInitialize() is Succeeded.\n");
		fclose(gpFile_ap);
	}

	//Game Loop
	while (bDone_ap == false)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
			{
				bDone_ap = true;
				break;
			}
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}
		}

		else
		{
			d3dDisplay();
			if (gbActiveWindow_ap == true)
			{
				if (gbEscapeKeyIsPressed_ap == true)
				{
					bDone_ap = true;
				}
				//update
				d3dUpdate();
			}

		}
	}

	//clean-up
	d3dUninitialise();

	return((int)msg.wParam);
}
LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	//function declaration
	void d3dToggleFullScreen(void);
	HRESULT d3dResize(int, int);
	void d3dUninitialise(void);

	//varible declaration
	HRESULT hr_ap;

	//code
	switch (iMsg)
	{

	case WM_ACTIVATE:
		if (HIWORD(wParam) == 0)
			gbActiveWindow_ap = true;
		else
			gbActiveWindow_ap = false;
		break;

	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			if (gbEscapeKeyIsPressed_ap == false)
				gbEscapeKeyIsPressed_ap = true;

			DestroyWindow(hwnd);
			break;
		}
		break;

	case WM_CHAR:
		switch (wParam)
		{
		case 'F':
		case 'f':
			d3dToggleFullScreen();
			break;
		}
		break;


	case WM_SIZE:
		if (gpID3D11DeviceContext)
		{
			hr_ap = d3dResize(LOWORD(lParam), HIWORD(lParam));
			if (FAILED(hr_ap))
			{
				fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
				fprintf_s(gpFile_ap, "d3dResize() is Succeeded.\n");
				fclose(gpFile_ap);
				return(hr_ap);
			}
			else
			{
				fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
				fprintf_s(gpFile_ap, "d3dResize() is Succeeded.\n");
				fclose(gpFile_ap);
			}
		}
		break;
	case WM_ERASEBKGND:
		return(0);
		break;
	case WM_CLOSE:
		DestroyWindow(hwnd);
		break;

	case WM_DESTROY:
		d3dUninitialise();
		PostQuitMessage(0);
		break;
	}
	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}
void d3dToggleFullScreen(void)
{
	//Variable declaration
	MONITORINFO mi;

	//code
	if (gbFullScreen_ap == FALSE)
	{
		dwStyle = GetWindowLong(ghwnd, GWL_STYLE);
		if (dwStyle & WS_OVERLAPPEDWINDOW)
		{
			mi = { sizeof(MONITORINFO) };
			if (GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi))
			{
				SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
				SetWindowPos(ghwnd,
					HWND_TOP,
					mi.rcMonitor.left,
					mi.rcMonitor.top,
					mi.rcMonitor.right - mi.rcMonitor.left,
					mi.rcMonitor.bottom - mi.rcMonitor.top,
					SWP_NOZORDER | SWP_FRAMECHANGED);
			}
		}
		ShowCursor(FALSE);
		gbFullScreen_ap = TRUE;
	}

	else
	{
		SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
		SetWindowPlacement(ghwnd, &wpPrev);
		SetWindowPos(ghwnd,
			HWND_TOP,
			0,
			0,
			0,
			0,
			SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER);
		ShowCursor(TRUE);
		gbFullScreen_ap = FALSE;
	}
}

HRESULT LoadD3DTexture(const wchar_t *textureFilename, ID3D11ShaderResourceView **ppID3D11ShaderResourceView)
{
	//variable declaration
	HRESULT hr;

	//code
	//create texture
	hr = DirectX::CreateWICTextureFromFile(	gpID3D11Device, 
											gpID3D11DeviceContext,
											textureFilename,
											NULL,
											ppID3D11ShaderResourceView);
	if (FAILED(hr))
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, " DirectX::CreateWICTextureFromFile() is Failed.\n");
		fclose(gpFile_ap);
		return(hr);
	}
	else
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, " DirectX::CreateWICTextureFromFile() is Succeeded.\n");
		fclose(gpFile_ap);
	}

	return(hr);
}

HRESULT d3dInitialise(void)
{
	//function declaration
	HRESULT d3dResize(int, int);
	void d3dUninitialise(void);

	//variable declaration
	HRESULT hr_ap;
	D3D_DRIVER_TYPE d3dDriverType;
	D3D_DRIVER_TYPE d3dDriverTypes[] =
	{
		D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_WARP, D3D_DRIVER_TYPE_REFERENCE
	};

	D3D_FEATURE_LEVEL d3dFeatureLevel_required = D3D_FEATURE_LEVEL_11_0;
	D3D_FEATURE_LEVEL d3dFeatureLevel_acquired = D3D_FEATURE_LEVEL_10_0;

	UINT createDeviceFlags = 0;
	UINT numDriverTypes = 0;
	UINT numFeatureLevels = 1;

	//code
	numDriverTypes = sizeof(d3dDriverTypes) / sizeof(d3dDriverTypes[0]);

	DXGI_SWAP_CHAIN_DESC dxgiSwapChainDesc;
	ZeroMemory((void *)&dxgiSwapChainDesc, sizeof(DXGI_SWAP_CHAIN_DESC));
	dxgiSwapChainDesc.BufferCount = 1;
	dxgiSwapChainDesc.BufferDesc.Width = WIN_WIDTH;
	dxgiSwapChainDesc.BufferDesc.Height = WIN_HEIGHT;
	dxgiSwapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	dxgiSwapChainDesc.BufferDesc.RefreshRate.Numerator = 60;
	dxgiSwapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
	dxgiSwapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	dxgiSwapChainDesc.OutputWindow = ghwnd;
	dxgiSwapChainDesc.SampleDesc.Count = 1;
	dxgiSwapChainDesc.SampleDesc.Quality = 0;
	dxgiSwapChainDesc.Windowed = TRUE;

	for (UINT driverTypeIndex = 0; driverTypeIndex < numDriverTypes; driverTypeIndex++)
	{
		d3dDriverType = d3dDriverTypes[driverTypeIndex];
		hr_ap = D3D11CreateDeviceAndSwapChain(
			NULL,									//	ADAPTER
			d3dDriverType,							//	DRIVER TYPE
			NULL,									//	SOFTWARE
			createDeviceFlags,						//	FLAGS
			&d3dFeatureLevel_required,				//	FEATURE LEVELS
			numFeatureLevels,						//	NUM FEATURE LEVELS
			D3D11_SDK_VERSION,						//	SDK VERSION
			&dxgiSwapChainDesc,						//	SWAP CHAIN DESC
			&gpIDXGISwapChain,						//	SWAP CHAIN
			&gpID3D11Device,						//	DEVICE
			&d3dFeatureLevel_acquired,				//	FEATURE LEVEL
			&gpID3D11DeviceContext);				//	DEVICE CONTEXT
		if (SUCCEEDED(hr_ap))
			break;
	}
	if (FAILED(hr_ap))
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "D3D11CreateDeviceAndSwapChain() is Failed.\n");
		fclose(gpFile_ap);
		return(hr_ap);
	}
	else
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "D3D11CreateDeviceAndSwapChain() is Succeeded.\n");
		fprintf_s(gpFile_ap, "The Choosen Driver Is Of");
		if (d3dDriverType == D3D_DRIVER_TYPE_HARDWARE)
		{
			fprintf_s(gpFile_ap, "Hardware Type.\n");
		}
		else if (d3dDriverType == D3D_DRIVER_TYPE_WARP)
		{
			fprintf_s(gpFile_ap, "Warp Type.\n");
		}
		else if (d3dDriverType == D3D_DRIVER_TYPE_REFERENCE)
		{
			fprintf_s(gpFile_ap, "Reference Type.\n");
		}
		else
		{
			fprintf_s(gpFile_ap, "Unknown Type.\n");
		}

		fprintf_s(gpFile_ap, "The Supported Highest Feature Level Is");
		if (d3dFeatureLevel_acquired == D3D_FEATURE_LEVEL_11_0)
		{
			fprintf_s(gpFile_ap, "11.0\n");
		}
		else if (d3dFeatureLevel_acquired == D3D_FEATURE_LEVEL_10_1)
		{
			fprintf_s(gpFile_ap, "10.1\n");
		}
		else if (d3dFeatureLevel_acquired == D3D_FEATURE_LEVEL_10_0)
		{
			fprintf_s(gpFile_ap, "10.0\n");
		}
		else
		{
			fprintf_s(gpFile_ap, "Unknown.\n");
		}
		fclose(gpFile_ap);
	}

	//initialise shader, input layouts, constant buffers
	const char *vertexShaderSourceCode =
		"cbuffer ConstantBuffer" \
		"{" \
		"float4x4 worldViewProjectionMatrix;" \
		"}" \
		"struct vertex_output" \
		"{" \
		"float4 position : SV_POSITION;" \
		"float2 texcoord : TEXCOORD;" \
		"};" \
		"vertex_output main(float4 pos:POSITION,float2 texcoord:TEXCOORD)" \
		"{" \
		"vertex_output output;" \
		"output.position = mul(worldViewProjectionMatrix, pos);" \
		"output.texcoord = texcoord;" \
		"return(output);" \
		"}";

	ID3DBlob *pID3DBlob_VertexShaderSourceCode = NULL;
	ID3DBlob *pID3DBlob_Error = NULL;

	hr_ap = D3DCompile(vertexShaderSourceCode,
		lstrlenA(vertexShaderSourceCode),
		"VS",
		NULL,
		D3D_COMPILE_STANDARD_FILE_INCLUDE,
		"main",
		"vs_5_0",
		0,
		0,
		&pID3DBlob_VertexShaderSourceCode,
		&pID3DBlob_Error);

	if (FAILED(hr_ap))
	{
		if (pID3DBlob_Error != NULL)
		{
			fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
			fprintf_s(gpFile_ap, "D3DCompile() is Failed For Vertex Shader :%s \n", (char *)pID3DBlob_Error->GetBufferPointer());
			fclose(gpFile_ap);
			pID3DBlob_Error->Release();
			pID3DBlob_Error = NULL;
			return(hr_ap);
		}
	}
	else
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "D3DCompile() is Succeeded For Vertex Shader.\n");
		fclose(gpFile_ap);
	}

	hr_ap = gpID3D11Device->CreateVertexShader(pID3DBlob_VertexShaderSourceCode->GetBufferPointer(),		//shader's binary code
		pID3DBlob_VertexShaderSourceCode->GetBufferSize(),													//size of that code
		NULL,																								//class linkage parameter
		&gpID3D11VertexShader);																				//Empty Utensil To Give The Vertex Shader Back

	if (FAILED(hr_ap))
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "CreateVertexShader() is Failed.\n");
		fclose(gpFile_ap);
		return(hr_ap);
	}
	else
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "gpID3D11Device::CreateVertexShader() is Succeeded.\n");
		fclose(gpFile_ap);
	}

	gpID3D11DeviceContext->VSSetShader(gpID3D11VertexShader, 0, 0);

	/* Pixel Shader */
	const char *pixelShaderSourceCode =
		"Texture2D myTexture2D;" \
		"SamplerState samplerState;" \
		"float4 main(float4 pos:SV_POSITION, float2 texcoord:TEXCOORD) : SV_TARGET" \
		"{" \
		"float4 color = myTexture2D.Sample(samplerState, texcoord);" \
		"return(color);" \
		"}";

	ID3DBlob *pID3DBlob_PixelShader = NULL;
	pID3DBlob_Error = NULL;

	hr_ap = D3DCompile(pixelShaderSourceCode,
		lstrlenA(pixelShaderSourceCode),
		"PS",
		NULL,
		D3D_COMPILE_STANDARD_FILE_INCLUDE,
		"main",
		"ps_5_0",
		0,
		0,
		&pID3DBlob_PixelShader,
		&pID3DBlob_Error);

	if (FAILED(hr_ap))
	{
		if (pID3DBlob_Error != NULL)
		{
			fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
			fprintf_s(gpFile_ap, "D3DCompile() is Failed For Pixel Shader :%s \n", (char *)pID3DBlob_Error->GetBufferPointer());
			fclose(gpFile_ap);
			pID3DBlob_Error->Release();
			pID3DBlob_Error = NULL;
			return(hr_ap);
		}
	}
	else
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "D3DCompile() is Succeeded For Pixel Shader.\n");
		fclose(gpFile_ap);
	}

	hr_ap = gpID3D11Device->CreatePixelShader(pID3DBlob_PixelShader->GetBufferPointer(),		//shader's binary code
		pID3DBlob_PixelShader->GetBufferSize(),													//size of that code
		NULL,																					//class linkage parameter
		&gpID3D11PixelShader);																	//Empty Utensil To Give The Vertex Shader Back

	if (FAILED(hr_ap))
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "CreateVertexShader() is Failed.\n");
		fclose(gpFile_ap);
		return(hr_ap);
	}
	else
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "gpID3D11Device::CreateVertexShader() is Succeeded.\n");
		fclose(gpFile_ap);
	}

	gpID3D11DeviceContext->PSSetShader(gpID3D11PixelShader, 0, 0);

	//pyramid vertices
	float pyramidVertices[] =
	{
		0.0f, 1.0f, 0.0f,	//front top
		-1.0f, -1.0f, 1.0f,	//front left
		1.0f, -1.0f, 1.0f,	//front right

		//triangle of right side
		0.0f, 1.0f, 0.0f,
		1.0f, -1.0f, 1.0f,
		1.0f, -1.0f, -1.0f,

		//triangle of back side
		0.0f, 1.0f, 0.0f,
		1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,

		//triangle of left side
		0.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f, 1.0f
	};

	//pyramid texcoords
	float pyramidTexcoords[] =
	{
		0.5f, 1.0f,		//front-top
		0.0f, 0.0f,		//front-left
		1.0f, 0.0f,		//front-right

		0.5f, 1.0f,		//right-top
		1.0f, 0.0f,		//right-left
		0.0f, 0.0f,		//right-right

		0.5f, 1.0f,		//back-top
		1.0f, 0.0f,		//back-left
		0.0f, 0.0f,		//back-right

		0.5f, 1.0f,		//left-top
		0.0f, 0.0f,		//left-left
		1.0f, 0.0f,		//left-right
	};


	//rectangle vertices
	float cubeVertices[] =
	{
		//Side-1
		-1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, -1.0f, 
		
		-1.0f, 1.0f, -1.0f,
		1.0f, 1.0f, 1.0f,
		1.0f, 1.0f, -1.0f,

		//side-2
		1.0f, -1.0f, -1.0f,
		1.0f, -1.0f, 1.0f,
		-1.0f, -1.0f, -1.0f,

		-1.0f, -1.0f, -1.0f,
		1.0f, -1.0f, 1.0f,
		-1.0f, -1.0f, 1.0f,

		//side-3
		-1.0f, 1.0f, -1.0f,
		1.0f, 1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,

		-1.0f, -1.0f, -1.0f,
		1.0f, 1.0f, -1.0f,
		1.0f, -1.0f, -1.0f,

		//side-4
		1.0f, -1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,
		-1.0f, -1.0f, 1.0f,

		-1.0f, -1.0f, 1.0f,
		1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, 1.0f,

		//side-5
		-1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, -1.0f, 1.0f,

		-1.0f, -1.0f, 1.0f,
		-1.0f, 1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,

		//side-6
		1.0f, -1.0f, -1.0f,
		1.0f, 1.0f, -1.0f,
		1.0f, -1.0f, 1.0f,

		1.0f, -1.0f, 1.0f,
		1.0f, 1.0f, -1.0f,
		1.0f, 1.0f, 1.0f
	};

	float cubeTexcoord[] = 
	{
		//SIDE 1 ( TOP )
		//triangle 1
		0.0f, 0.0f, 
		0.0f, 1.0f, 
		1.0f, 0.0f, 
		//triangle 2
		1.0f, 0.0f, 
		0.0f, 1.0f, 
		1.0f, 1.0f, 

		//SIDE 2 ( BOTTOM )
		//triangle 1
		0.0f, 0.0f, 
		0.0f, 1.0f,
		1.0f, 0.0f, 
		//triangle 2
		1.0f, 0.0f, 
		0.0f, 1.0f, 
		1.0f, 1.0f, 

		//SIDE 3 ( FRONT )
		//triangle 1
		0.0f, 0.0f, 
		0.0f, 1.0f,
		1.0f, 0.0f, 
		//triangle 2
		1.0f, 0.0f, 
		0.0f, 1.0f, 
		1.0f, 1.0f, 

		//SIDE 4 ( BACK )
		//triangle 1
		0.0f, 0.0f, 
		0.0f, 1.0f,
		1.0f, 0.0f, 
		//triangle 2
		1.0f, 0.0f, 
		0.0f, 1.0f, 
		1.0f, 1.0f, 

		//SIDE 5 ( LEFT )
		//triangle 1
		0.0f, 0.0f, 
		0.0f, 1.0f,
		1.0f, 0.0f, 
		//triangle 2
		1.0f, 0.0f, 
		0.0f, 1.0f, 
		1.0f, 1.0f, 

		//SIDE 6 ( RIGHT )
		//triangle 1
		0.0f, 0.0f, 
		0.0f, 1.0f,
		1.0f, 0.0f, 
		//triangle 2
		1.0f, 0.0f, 
		0.0f, 1.0f, 
		1.0f, 1.0f, 
	};



	//create and set input layout
	D3D11_INPUT_ELEMENT_DESC inputElementDesc[2];

	/*Position*/

	inputElementDesc[0].SemanticName = "POSITION";
	inputElementDesc[0].SemanticIndex = 0;
	inputElementDesc[0].Format = DXGI_FORMAT_R32G32B32_FLOAT;
	inputElementDesc[0].InputSlot = 0;
	inputElementDesc[0].AlignedByteOffset = 0;
	inputElementDesc[0].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	inputElementDesc[0].InstanceDataStepRate = 0;

	/* TexCoords */

	inputElementDesc[1].SemanticName = "TEXCOORD";
	inputElementDesc[1].SemanticIndex = 0;
	inputElementDesc[1].Format = DXGI_FORMAT_R32G32_FLOAT;
	inputElementDesc[1].InputSlot = 1;
	inputElementDesc[1].AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
	inputElementDesc[1].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	inputElementDesc[1].InstanceDataStepRate = 0;

	hr_ap = gpID3D11Device->CreateInputLayout(inputElementDesc,
		_ARRAYSIZE(inputElementDesc),
		pID3DBlob_VertexShaderSourceCode->GetBufferPointer(),
		pID3DBlob_VertexShaderSourceCode->GetBufferSize(),
		&gpID3D11InputLayout);

	if (FAILED(hr_ap))
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "gpID3D11Device::CreateInputLayout() is Failed.\n");
		fclose(gpFile_ap);
		return(hr_ap);
	}
	else
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "gpID3D11Device::CreateInputLayout() is Succeeded.\n");
		fclose(gpFile_ap);
	}
	gpID3D11DeviceContext->IASetInputLayout(gpID3D11InputLayout);

	//release vertex shader
	pID3DBlob_VertexShaderSourceCode->Release();
	pID3DBlob_VertexShaderSourceCode = NULL;

	//release pixel shader
	pID3DBlob_PixelShader->Release();
	pID3DBlob_PixelShader = NULL;

	/* PYRAMID */

	//position

	//create vertex buffer
	D3D11_BUFFER_DESC bufferDescPyramidPosition;
	ZeroMemory(&bufferDescPyramidPosition, sizeof(D3D11_BUFFER_DESC));
	bufferDescPyramidPosition.Usage = D3D11_USAGE_DYNAMIC;
	bufferDescPyramidPosition.ByteWidth = sizeof(float) * ARRAYSIZE(pyramidVertices);
	bufferDescPyramidPosition.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDescPyramidPosition.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	hr_ap = gpID3D11Device->CreateBuffer(&bufferDescPyramidPosition,
		NULL,
		&gpID3D11Buffer_VertexBuffer_Pyramid_Position);

	if (FAILED(hr_ap))
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "gpID3D11Device::CreateBuffer() is Failed.\n");
		fclose(gpFile_ap);
		return(hr_ap);
	}
	else
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "gpID3D11Device::CreateBuffer() is Succeeded.\n");
		fclose(gpFile_ap);
	}

	//copy vertices into above buffer
	D3D11_MAPPED_SUBRESOURCE mappedSubresourcePyramidPosition;
	ZeroMemory(&mappedSubresourcePyramidPosition, sizeof(D3D11_MAPPED_SUBRESOURCE));
	gpID3D11DeviceContext->Map(gpID3D11Buffer_VertexBuffer_Pyramid_Position,
		NULL,
		D3D11_MAP_WRITE_DISCARD,
		NULL,
		&mappedSubresourcePyramidPosition);
	memcpy(mappedSubresourcePyramidPosition.pData, pyramidVertices, sizeof(pyramidVertices));
	gpID3D11DeviceContext->Unmap(gpID3D11Buffer_VertexBuffer_Pyramid_Position, NULL);

	//texture

	//create vertex buffer
	D3D11_BUFFER_DESC bufferDescPyramidTexture;
	ZeroMemory(&bufferDescPyramidTexture, sizeof(D3D11_BUFFER_DESC));
	bufferDescPyramidTexture.Usage = D3D11_USAGE_DYNAMIC;
	bufferDescPyramidTexture.ByteWidth = sizeof(float) * _ARRAYSIZE(pyramidTexcoords);
	bufferDescPyramidTexture.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDescPyramidTexture.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	hr_ap = gpID3D11Device->CreateBuffer(&bufferDescPyramidTexture, NULL,
		&gpID3D11Buffer_VertexBuffer_Pyramid_Texture);
	if (FAILED(hr_ap))
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "gpID3D11Device::CreateBuffer() is Failed.\n");
		fclose(gpFile_ap);
		return(hr_ap);
	}
	else
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "gpID3D11Device::CreateBuffer() is Succeeded.\n");
		fclose(gpFile_ap);
	}

	//copy vertices into above buffer
	D3D11_MAPPED_SUBRESOURCE mappedSubresourcePyramidTexture;
	ZeroMemory(&mappedSubresourcePyramidTexture, sizeof(D3D11_MAPPED_SUBRESOURCE));
	gpID3D11DeviceContext->Map(gpID3D11Buffer_VertexBuffer_Pyramid_Texture,
		NULL,
		D3D11_MAP_WRITE_DISCARD,
		NULL,
		&mappedSubresourcePyramidTexture);
	memcpy(mappedSubresourcePyramidTexture.pData, pyramidTexcoords, sizeof(pyramidTexcoords));
	gpID3D11DeviceContext->Unmap(gpID3D11Buffer_VertexBuffer_Pyramid_Texture, NULL);


	/* CUBE */

	//position

	//create vertex buffer
	D3D11_BUFFER_DESC bufferDescCubePosition;
	ZeroMemory(&bufferDescCubePosition, sizeof(D3D11_BUFFER_DESC));
	bufferDescCubePosition.Usage = D3D11_USAGE_DYNAMIC;
	bufferDescCubePosition.ByteWidth = sizeof(float) * ARRAYSIZE(cubeVertices);
	bufferDescCubePosition.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDescCubePosition.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	hr_ap = gpID3D11Device->CreateBuffer(&bufferDescCubePosition, NULL,
		&gpID3D11Buffer_VertexBuffer_Cube_Position);
	if (FAILED(hr_ap))
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "gpID3D11Device::CreateBuffer() is Failed.\n");
		fclose(gpFile_ap);
		return(hr_ap);
	}
	else
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "gpID3D11Device::CreateBuffer() is Succeeded.\n");
		fclose(gpFile_ap);
	}

	//copy vertices into above buffer
	D3D11_MAPPED_SUBRESOURCE mappedSubresourceCubePosition;
	ZeroMemory(&mappedSubresourceCubePosition, sizeof(D3D11_MAPPED_SUBRESOURCE));
	gpID3D11DeviceContext->Map(gpID3D11Buffer_VertexBuffer_Cube_Position,
		NULL,
		D3D11_MAP_WRITE_DISCARD,
		NULL,
		&mappedSubresourceCubePosition);
	memcpy(mappedSubresourceCubePosition.pData, cubeVertices, sizeof(cubeVertices));
	gpID3D11DeviceContext->Unmap(gpID3D11Buffer_VertexBuffer_Cube_Position, NULL);

	//texture

	//create vertex buffer
	D3D11_BUFFER_DESC bufferDescCubeTexture;
	ZeroMemory(&bufferDescCubeTexture, sizeof(D3D11_BUFFER_DESC));
	bufferDescCubeTexture.Usage = D3D11_USAGE_DYNAMIC;
	bufferDescCubeTexture.ByteWidth = sizeof(float) * _ARRAYSIZE(cubeTexcoord);
	bufferDescCubeTexture.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	bufferDescCubeTexture.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

	hr_ap = gpID3D11Device->CreateBuffer(&bufferDescCubeTexture, NULL,
		&gpID3D11Buffer_VertexBuffer_Cube_Texture);
	if (FAILED(hr_ap))
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "gpID3D11Device::CreateBuffer() is Failed.\n");
		fclose(gpFile_ap);
		return(hr_ap);
	}
	else
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "gpID3D11Device::CreateBuffer() is Succeeded.\n");
		fclose(gpFile_ap);
	}

	//copy vertices into above buffer
	D3D11_MAPPED_SUBRESOURCE mappedSubresourceCubeTexture;
	ZeroMemory(&mappedSubresourceCubeTexture, sizeof(D3D11_MAPPED_SUBRESOURCE));
	gpID3D11DeviceContext->Map(gpID3D11Buffer_VertexBuffer_Cube_Texture,
		NULL,
		D3D11_MAP_WRITE_DISCARD,
		NULL,
		&mappedSubresourceCubeTexture);
	memcpy(mappedSubresourceCubeTexture.pData, cubeTexcoord, sizeof(cubeTexcoord));
	gpID3D11DeviceContext->Unmap(gpID3D11Buffer_VertexBuffer_Cube_Texture, NULL);

	
	//define constant buffer
	D3D11_BUFFER_DESC bufferDesc_ConstantBuffer;
	ZeroMemory(&bufferDesc_ConstantBuffer, sizeof(D3D11_BUFFER_DESC));
	bufferDesc_ConstantBuffer.Usage = D3D11_USAGE_DEFAULT;
	bufferDesc_ConstantBuffer.ByteWidth = sizeof(CBUFFER);
	bufferDesc_ConstantBuffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

	hr_ap = gpID3D11Device->CreateBuffer(&bufferDesc_ConstantBuffer, nullptr,
		&gpID3D11Buffer_ConstantBuffer);
	if (FAILED(hr_ap))
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "gpID3D11Device::CreateBuffer() is Failed for constant buffer.\n");
		fclose(gpFile_ap);
		return(hr_ap);
	}
	else
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "gpID3D11Device::CreateBuffer() is Succeeded for constant buffer.\n");
		fclose(gpFile_ap);
	}

	gpID3D11DeviceContext->VSSetConstantBuffers(0, 1, &gpID3D11Buffer_ConstantBuffer);

	D3D11_RASTERIZER_DESC rasterizerDesc;
	ZeroMemory((void *)&rasterizerDesc, sizeof(D3D11_RASTERIZER_DESC));
	rasterizerDesc.AntialiasedLineEnable = FALSE;
	rasterizerDesc.CullMode = D3D11_CULL_NONE;
	rasterizerDesc.DepthBias = 0;
	rasterizerDesc.DepthBiasClamp = 0.0f;
	rasterizerDesc.DepthClipEnable = TRUE;
	rasterizerDesc.FillMode = D3D11_FILL_SOLID;
	rasterizerDesc.FrontCounterClockwise = FALSE;
	rasterizerDesc.ScissorEnable = FALSE;
	rasterizerDesc.SlopeScaledDepthBias = 0.0f;
	hr_ap = gpID3D11Device->CreateRasterizerState(&rasterizerDesc, &gpID3D11RasterizerState);
	if (FAILED(hr_ap))
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "gpID3D11Device::CreateRasterizerState() is Failed for constant buffer.\n");
		fclose(gpFile_ap);
		return(hr_ap);
	}
	else
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "gpID3D11Device::CreateRasterizerState() is Succeeded for constant buffer.\n");
		fclose(gpFile_ap);
	}

	gpID3D11DeviceContext->RSSetState(gpID3D11RasterizerState);

	//PYRAMID
	//create texture
	hr_ap = LoadD3DTexture(L"Stone.bmp", &gpID3D11ShaderResourceView_Pyramid_Texture);
	if (FAILED(hr_ap))
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "LoadD3DTexture() is Failed for pyramid.\n");
		fclose(gpFile_ap);
		return(hr_ap);
	}
	else
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "LoadD3DTexture() is Succeeded for pyramid.\n");
		fclose(gpFile_ap);
	}	

	//create the sample state
	D3D11_SAMPLER_DESC samplerDesc_pyramid;
	ZeroMemory(&samplerDesc_pyramid, sizeof(D3D11_SAMPLER_DESC));
	samplerDesc_pyramid.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	samplerDesc_pyramid.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc_pyramid.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc_pyramid.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;

	hr_ap = gpID3D11Device->CreateSamplerState(&samplerDesc_pyramid, &gpID3D11SamplerState_Pyramid_Texture);
	if (FAILED(hr_ap))
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "gpID3D11Device->CreateSamplerState() is Failed for pyramid.\n");
		fclose(gpFile_ap);
		return(hr_ap);
	}
	else
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "gpID3D11Device->CreateSamplerState() is Succeeded for pyramid.\n");
		fclose(gpFile_ap);
	}	

	//CUBE
	//create texture
	hr_ap = LoadD3DTexture(L"Kundali.bmp", &gpID3D11ShaderResourceView_Cube_Texture);
	if (FAILED(hr_ap))
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "LoadD3DTexture() is Failed for pyramid.\n");
		fclose(gpFile_ap);
		return(hr_ap);
	}
	else
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "LoadD3DTexture() is Succeeded for pyramid.\n");
		fclose(gpFile_ap);
	}	

	//create the sample state
	D3D11_SAMPLER_DESC samplerDesc_cube;
	ZeroMemory(&samplerDesc_cube, sizeof(D3D11_SAMPLER_DESC));
	samplerDesc_cube.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	samplerDesc_cube.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc_cube.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc_cube.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;

	hr_ap = gpID3D11Device->CreateSamplerState(&samplerDesc_cube, &gpID3D11SamplerState_Cube_Texture);
	if (FAILED(hr_ap))
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "gpID3D11Device->CreateSamplerState() is Failed for pyramid.\n");
		fclose(gpFile_ap);
		return(hr_ap);
	}
	else
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "gpID3D11Device->CreateSamplerState() is Succeeded for pyramid.\n");
		fclose(gpFile_ap);
	}	


	//d3d clear color
	gClearColor[0] = 0.0f;
	gClearColor[1] = 0.0f;
	gClearColor[2] = 0.0f;
	gClearColor[3] = 1.0f;

	//identity projection matrix
	gPerspectiveProjectionMatrix = XMMatrixIdentity();

	//call resize for first time
	hr_ap = d3dResize(WIN_WIDTH, WIN_HEIGHT);
	if (FAILED(hr_ap))
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "d3dResize() is Succeeded.\n");
		fclose(gpFile_ap);
		return(hr_ap);
	}
	else
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "d3dResize() is Succeeded.\n");
		fclose(gpFile_ap);
	}

	return(S_OK);
}
HRESULT d3dResize(int width, int height)
{
	//code
	HRESULT hr_ap = S_OK;

	//free any size-dependant resource
	if (gpID3D11RenderTargetView)
	{
		gpID3D11RenderTargetView->Release();
		gpID3D11RenderTargetView = NULL;
	}

	if (gpID3D11DepthStencilView)
	{
		gpID3D11DepthStencilView->Release();
		gpID3D11DepthStencilView = NULL;
	}

	//resize swap chain buffers accordingly
	gpIDXGISwapChain->ResizeBuffers(1, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, 0);

	//again get back buffer from swap chain
	ID3D11Texture2D *pID3D11Texture2D_BackBuffer;
	gpIDXGISwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID *)&pID3D11Texture2D_BackBuffer);

	//again get render target view from d3d11 device using above back buffer
	hr_ap = gpID3D11Device->CreateRenderTargetView(pID3D11Texture2D_BackBuffer, NULL, &gpID3D11RenderTargetView);
	if (FAILED(hr_ap))
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "gpID3D11Device::CreateRenderTargetView is Failed.\n");
		fclose(gpFile_ap);
		return(hr_ap);
	}
	else
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "gpID3D11Device::CreateRenderTargetView is Succeeded.\n");
		fclose(gpFile_ap);
	}
	pID3D11Texture2D_BackBuffer->Release();
	pID3D11Texture2D_BackBuffer = NULL;

	//create depth stencil buffer
	D3D11_TEXTURE2D_DESC textureDesc;
	ZeroMemory(&textureDesc, sizeof(D3D11_TEXTURE2D_DESC));
	textureDesc.Width = (UINT)width;
	textureDesc.Height = (UINT)height;
	textureDesc.ArraySize = 1;
	textureDesc.MipLevels = 1;
	textureDesc.SampleDesc.Count = 1;	//in real world this can be upto 4.
	textureDesc.SampleDesc.Quality = 0;	//if above is 4 then it is 1.
	textureDesc.Format = DXGI_FORMAT_D32_FLOAT;	//openGL's 24 bit depth In D3D It Is 32 bit.
	textureDesc.Usage = D3D11_USAGE_DEFAULT;
	textureDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
	textureDesc.CPUAccessFlags = 0;
	textureDesc.MiscFlags = 0;
	ID3D11Texture2D *pID3D11Texture2D_DepthBuffer;
	gpID3D11Device->CreateTexture2D(&textureDesc, NULL, &pID3D11Texture2D_DepthBuffer);

	//create depth stencil buffer from above depth stencil buffer
	D3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc;
	ZeroMemory(&depthStencilViewDesc, sizeof(D3D11_DEPTH_STENCIL_VIEW_DESC));
	depthStencilViewDesc.Format = DXGI_FORMAT_D32_FLOAT;
	depthStencilViewDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMS;		//MS For MultiSample
	hr_ap = gpID3D11Device->CreateDepthStencilView(pID3D11Texture2D_DepthBuffer, 
		&depthStencilViewDesc,
		&gpID3D11DepthStencilView);

	if (FAILED(hr_ap))
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "gpID3D11Device::CreateDepthStencilView() is Failed for depth buffer.\n");
		fclose(gpFile_ap);
		return(hr_ap);
	}
	else
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "gpID3D11Device::CreateDepthStencilView() is Succeeded for depth buffer.\n");
		fclose(gpFile_ap);
	}

	pID3D11Texture2D_DepthBuffer->Release();
	pID3D11Texture2D_DepthBuffer = NULL;

	//set render target view as render target
	gpID3D11DeviceContext->OMSetRenderTargets(1, &gpID3D11RenderTargetView, gpID3D11DepthStencilView);

	//set viewport
	D3D11_VIEWPORT d3dViewPort;
	d3dViewPort.TopLeftX = 0;
	d3dViewPort.TopLeftY = 0;
	d3dViewPort.Width = (float)width;
	d3dViewPort.Height = (float)height;
	d3dViewPort.MinDepth = 0.0f;
	d3dViewPort.MaxDepth = 1.0f;
	gpID3D11DeviceContext->RSSetViewports(1, &d3dViewPort);

	//set perspective matrix
	gPerspectiveProjectionMatrix = XMMatrixPerspectiveFovLH(XMConvertToRadians(45.0f), ((float)width / (float)height), 0.1f, 100.0f);

	return(hr_ap);
}
void d3dUpdate(void)
{
	RotationAngle = RotationAngle + 0.005f;
	if (RotationAngle > 360.0f)
	{
		RotationAngle = 0.0f;
	}
}
void d3dDisplay(void)
{
	//code
	//clear render target view to a chosen color
	gpID3D11DeviceContext->ClearRenderTargetView(gpID3D11RenderTargetView, gClearColor);

	//clear depth stencil vies
	gpID3D11DeviceContext->ClearDepthStencilView(gpID3D11DepthStencilView,
		D3D11_CLEAR_DEPTH,
		1.0f,
		0);

	/* Pyramid */

	//select which vertex buffer to display
	UINT stride = sizeof(float) * 3;
	UINT offset = 0;
	gpID3D11DeviceContext->IASetVertexBuffers(0, 1, &gpID3D11Buffer_VertexBuffer_Pyramid_Position, &stride, &offset);

	stride = sizeof(float) * 2;	//	2 is for u, v
	offset = 0;
	gpID3D11DeviceContext->IASetVertexBuffers(1, 1, &gpID3D11Buffer_VertexBuffer_Pyramid_Texture, &stride, &offset);

	//bind texture sampler as pixel shader resource
	gpID3D11DeviceContext->PSSetShaderResources(	0,
													1,
													&gpID3D11ShaderResourceView_Pyramid_Texture);
	gpID3D11DeviceContext->PSSetSamplers(	0,
											1,
											&gpID3D11SamplerState_Pyramid_Texture);

	//select primitive
	gpID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	//translation corncerned with world co-ordinates
	XMMATRIX worldMatrix = XMMatrixIdentity();
	XMMATRIX viewMatrix = XMMatrixIdentity();

	//transformation
	XMMATRIX translationMatrix = XMMatrixTranslation(-1.5f, 0.0f, 6.0f);
	XMMATRIX rotationMatrix = XMMatrixRotationY(-RotationAngle);
	worldMatrix = rotationMatrix * translationMatrix;

	//final worldViewProjection Matrix
	XMMATRIX wvpMatrix = worldMatrix * viewMatrix * gPerspectiveProjectionMatrix;

	//load the data into constant buffer
	CBUFFER constantBuffer_pyramid;
	constantBuffer_pyramid.WorldViewProjectionMatrix = wvpMatrix;

	gpID3D11DeviceContext->UpdateSubresource(gpID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer_pyramid, 0, 0);

	//draw vertex buffer
	gpID3D11DeviceContext->Draw(12, 0);

	/* Cube */
	//select which vertex buffer to display
	stride = sizeof(float) * 3;
	offset = 0;
	gpID3D11DeviceContext->IASetVertexBuffers(0, 1, &gpID3D11Buffer_VertexBuffer_Cube_Position, &stride, &offset);

	stride = sizeof(float) * 2;
	offset = 0;
	gpID3D11DeviceContext->IASetVertexBuffers(1, 1, &gpID3D11Buffer_VertexBuffer_Cube_Texture, &stride, &offset);

	//bind texture sampler as pixel shader resource
	gpID3D11DeviceContext->PSSetShaderResources(	0,
													1,
													&gpID3D11ShaderResourceView_Cube_Texture);
	gpID3D11DeviceContext->PSSetSamplers(	0,
											1,
											&gpID3D11SamplerState_Cube_Texture);


	//select primitive
	gpID3D11DeviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	//translation corncerned with world co-ordinates
	worldMatrix = XMMatrixIdentity();
	viewMatrix = XMMatrixIdentity();
	translationMatrix = XMMatrixIdentity();
	rotationMatrix = XMMatrixIdentity();
	wvpMatrix = XMMatrixIdentity();
	XMMATRIX scaleMatrix = XMMatrixScaling(0.75f, 0.75f, 0.75f);

	//transformation
	translationMatrix = XMMatrixTranslation(1.5f, 0.0f, 6.0f);

	//All Axis Rotation
	XMMATRIX rotation_X = XMMatrixRotationX(RotationAngle);
	XMMATRIX rotation_Y = XMMatrixRotationY(RotationAngle);
	XMMATRIX rotation_Z = XMMatrixRotationZ(RotationAngle);

	rotationMatrix = rotation_X * rotation_Y * rotation_Z;
	worldMatrix = scaleMatrix * rotationMatrix * translationMatrix;

	//final worldViewProjection Matrix
	wvpMatrix = worldMatrix * viewMatrix * gPerspectiveProjectionMatrix;

	//load the data into constant buffer
	CBUFFER constantBuffer_cube;
	constantBuffer_cube.WorldViewProjectionMatrix = wvpMatrix;

	gpID3D11DeviceContext->UpdateSubresource(gpID3D11Buffer_ConstantBuffer, 0, NULL, &constantBuffer_cube, 0, 0);

	//draw vertex buffer
	gpID3D11DeviceContext->Draw(6, 0);	//6 vertices from 0 to 6
	gpID3D11DeviceContext->Draw(6, 6);	//6 vertices from 6 to 12
	gpID3D11DeviceContext->Draw(6, 12);	//6 vertices from 12 to 18
	gpID3D11DeviceContext->Draw(6, 18);	//6 vertices from 18 to 24
	gpID3D11DeviceContext->Draw(6, 24);	//6 vertices from 24 to 30
	gpID3D11DeviceContext->Draw(6, 30);	//6 vertices from 30 to 36

	//switch between front and back
	gpIDXGISwapChain->Present(0, 0);
}
void d3dUninitialise(void)
{
	//code
	if (gpID3D11Buffer_ConstantBuffer)
	{
		gpID3D11Buffer_ConstantBuffer->Release();
		gpID3D11Buffer_ConstantBuffer = NULL;
	}

	if (gpID3D11InputLayout)
	{
		gpID3D11InputLayout->Release();
		gpID3D11InputLayout = NULL;
	}

	if (gpID3D11Buffer_VertexBuffer_Pyramid_Position)
	{
		gpID3D11Buffer_VertexBuffer_Pyramid_Position->Release();
		gpID3D11Buffer_VertexBuffer_Pyramid_Position = NULL;
	}

	if (gpID3D11Buffer_VertexBuffer_Pyramid_Texture)
	{
		gpID3D11Buffer_VertexBuffer_Pyramid_Texture->Release();
		gpID3D11Buffer_VertexBuffer_Pyramid_Texture = NULL;
	}

	if(gpID3D11ShaderResourceView_Pyramid_Texture)
	{
		gpID3D11ShaderResourceView_Pyramid_Texture->Release();
		gpID3D11ShaderResourceView_Pyramid_Texture = NULL;
	}

	if(gpID3D11SamplerState_Pyramid_Texture)
	{
		gpID3D11SamplerState_Pyramid_Texture->Release();
		gpID3D11SamplerState_Pyramid_Texture = NULL;
	}

	if (gpID3D11Buffer_VertexBuffer_Cube_Position)
	{
		gpID3D11Buffer_VertexBuffer_Cube_Position->Release();
		gpID3D11Buffer_VertexBuffer_Cube_Position = NULL;
	}

	if (gpID3D11Buffer_VertexBuffer_Cube_Texture)
	{
		gpID3D11Buffer_VertexBuffer_Cube_Texture->Release();
		gpID3D11Buffer_VertexBuffer_Cube_Texture = NULL;
	}

	if(gpID3D11ShaderResourceView_Cube_Texture)
	{
		gpID3D11ShaderResourceView_Cube_Texture->Release();
		gpID3D11ShaderResourceView_Cube_Texture = NULL;
	}

	if(gpID3D11SamplerState_Cube_Texture)
	{
		gpID3D11SamplerState_Cube_Texture->Release();
		gpID3D11SamplerState_Cube_Texture = NULL;
	}

	if (gpID3D11RasterizerState)
	{
		gpID3D11RasterizerState->Release();
		gpID3D11RasterizerState = NULL;
	}

	if (gpID3D11PixelShader)
	{
		gpID3D11PixelShader->Release();
		gpID3D11PixelShader = NULL;
	}

	if (gpID3D11VertexShader)
	{
		gpID3D11VertexShader->Release();
		gpID3D11VertexShader = NULL;
	}

	if (gpID3D11DepthStencilView)
	{
		gpID3D11DepthStencilView->Release();
		gpID3D11DepthStencilView = NULL;
	}

	if (gpID3D11RenderTargetView)
	{
		gpID3D11RenderTargetView->Release();
		gpID3D11RenderTargetView = NULL;
	}

	if (gpIDXGISwapChain)
	{
		gpIDXGISwapChain->Release();
		gpIDXGISwapChain = NULL;
	}

	if (gpID3D11DeviceContext)
	{
		gpID3D11DeviceContext->Release();
		gpID3D11DeviceContext = NULL;
	}

	if (gpID3D11Device)
	{
		gpID3D11Device->Release();
		gpID3D11Device = NULL;
	}

	if (gpFile_ap)
	{
		fopen_s(&gpFile_ap, gszLogFileName_ap, "a+");
		fprintf_s(gpFile_ap, "d3dUninitialise() is Succeeded.\n");
		fprintf_s(gpFile_ap, "Log File Is Successfully Closed.\n");
		fclose(gpFile_ap);
	}
}
