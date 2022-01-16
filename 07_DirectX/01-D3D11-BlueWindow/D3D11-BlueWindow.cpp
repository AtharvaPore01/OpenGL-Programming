//header files
#include <Windows.h>
#include <stdio.h>
#include <d3d11.h>

//libraries
#pragma comment (lib, "d3d11.lib")

#define WIN_WIDTH	800
#define WIN_HEIGHT	600
#define X (GetSystemMetrics(SM_CXSCREEN) - WIN_WIDTH)/2
#define Y (GetSystemMetrics(SM_CYSCREEN) - WIN_HEIGHT)/2

//global function declaration
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

//global variable declaration
FILE *gpFile_ap = NULL;
char gszLogFileName_ap[] = "Log.txt";

HWND ghwnd = NULL;

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
		TEXT("Direct3D11 : Blue Window"),
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
		hr_ap = D3D11CreateDeviceAndSwapChain(NULL,
			d3dDriverType,
			NULL,
			createDeviceFlags,
			&d3dFeatureLevel_required,
			numFeatureLevels,
			D3D11_SDK_VERSION,
			&dxgiSwapChainDesc,
			&gpIDXGISwapChain,
			&gpID3D11Device,
			&d3dFeatureLevel_acquired,
			&gpID3D11DeviceContext);
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

	//d3d clear color
	gClearColor[0] = 0.0f;
	gClearColor[1] = 0.0f;
	gClearColor[2] = 1.0f;
	gClearColor[3] = 1.0f;

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

	//set render target view as render target
	gpID3D11DeviceContext->OMSetRenderTargets(1, &gpID3D11RenderTargetView, NULL);

	//set viewport
	D3D11_VIEWPORT d3dViewPort;
	d3dViewPort.TopLeftX = 0;
	d3dViewPort.TopLeftY = 0;
	d3dViewPort.Width = (float)width;
	d3dViewPort.Height = (float)height;
	d3dViewPort.MinDepth = 0.0f;
	d3dViewPort.MaxDepth = 1.0f;
	gpID3D11DeviceContext->RSSetViewports(1, &d3dViewPort);

	return(hr_ap);
}
void d3dDisplay(void)
{
	//code
	//clear render target view to a chosen color
	gpID3D11DeviceContext->ClearRenderTargetView(gpID3D11RenderTargetView, gClearColor);

	//switch between front & back buffers
	gpIDXGISwapChain->Present(0, 0);
}
void d3dUninitialise(void)
{
	//code
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
