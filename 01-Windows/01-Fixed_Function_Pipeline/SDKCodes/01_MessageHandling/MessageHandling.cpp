//Header
#include<Windows.h>

//Function Decalaration
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

//WinMain
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
	WNDCLASSEX wndclass;
	HWND hwnd;
	MSG msg;
	TCHAR szAppName[] = TEXT("MyApp");

	wndclass.cbSize = sizeof(WNDCLASSEX);
	wndclass.style = CS_HREDRAW | CS_VREDRAW;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.lpfnWndProc = WndProc;
	wndclass.hInstance = hInstance;
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wndclass.lpszClassName = szAppName;
	wndclass.lpszMenuName = NULL;

	//Register Class
	RegisterClassEx(&wndclass);

	//CreateWindow
	hwnd = CreateWindow(szAppName,
		TEXT("Message Handling Window"),
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT,
		CW_USEDEFAULT,
		CW_USEDEFAULT,
		CW_USEDEFAULT,
		NULL,
		NULL,
		hInstance,
		NULL);

	ShowWindow(hwnd, iCmdShow);
	UpdateWindow(hwnd);

	//Message Loop
	while (GetMessage(&msg, NULL, 0, 0))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
	return((int)msg.wParam);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	switch (iMsg)
	{
	case WM_CREATE:
		MessageBox(hwnd, TEXT("This Is WM_CREATE Message"), TEXT("WM_CREATE"), MB_OK | MB_ICONINFORMATION);
		break;
	
	case WM_KEYDOWN:
		switch (wParam)
		{
		case VK_ESCAPE:
			MessageBox(hwnd, TEXT("This Is VK_ESCAPE Message"), TEXT("VK_ESCAPE"), MB_OK | MB_ICONINFORMATION);
			DestroyWindow(hwnd);
			break;
		case 0x46:
			MessageBox(hwnd, TEXT("This Is F Key Message"), TEXT("F Key"), MB_OK | MB_ICONINFORMATION);
			break;
		}
		break;

	case WM_LBUTTONDOWN:
		MessageBox(hwnd, TEXT("This Is WM_LBUTTONDOWN Message"), TEXT("WM_LBUTTONDOWN"), MB_OK | MB_ICONINFORMATION);
		break;

	case WM_RBUTTONDOWN:
		MessageBox(hwnd, TEXT("This Is WM_RBUTTONDOWN Message"), TEXT("WM_RBUTTONDOWN"), MB_OK | MB_ICONINFORMATION);
		break;

	case WM_DESTROY:
		MessageBox(hwnd, TEXT("This Is WM_DESTROY Message"), TEXT("WM_DESTROY"), MB_OK | MB_ICONINFORMATION);
		PostQuitMessage(0);
		break;
	}
	return(DefWindowProc(hwnd, iMsg, wParam, lParam));
}