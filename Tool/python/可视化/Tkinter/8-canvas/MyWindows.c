/* -------------------------------------------------------------------
                    MyWindows.c -- 基本窗口模型  
				《Windows 程序设计（SDK）》视频教程                    
--------------------------------------------------------------------*/

#include <windows.h>

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR szCmdLine, int iCmdShow)
{
	static TCHAR szAppName[] = TEXT("MyWindows");
	HWND hwnd;
	MSG msg;
	WNDCLASS wndclass;

	wndclass.style = CS_HREDRAW | CS_VREDRAW;
	wndclass.lpfnWndProc = WndProc;
	wndclass.cbClsExtra = 0;
	wndclass.cbWndExtra = 0;
	wndclass.hInstance = hInstance;
	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
	wndclass.lpszMenuName = NULL;
	wndclass.lpszClassName = szAppName;

	if (!RegisterClass(&wndclass))
	{
		MessageBox(NULL, TEXT("这个程序需要在 Windows NT 才能执行！"), szAppName, MB_ICONERROR);
		return 0;
	}

	hwnd = CreateWindow(szAppName, 
		TEXT("鱼C工作室"), 
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

	while (GetMessage(&msg, NULL, 0, 0))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	return msg.wParam;
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	HDC hdc;
	PAINTSTRUCT ps;
	HPEN hPen, hOldPen;
	HBRUSH hBlueBrush, hRedBrush, hYellowBrush, hOldBrush;
	POINT apt[128];
	static int cxClient, cyClient;

	switch (message)
	{
	case WM_SIZE:
		cxClient = LOWORD(lParam);
		cyClient = HIWORD(lParam);
		return 0;

	case WM_PAINT:
		hdc = BeginPaint(hwnd, &ps);
		
		// 辅助线
		hPen = CreatePen(PS_DOT, 1, RGB(192, 192, 192));
		hOldPen = SelectObject(hdc, hPen);
		MoveToEx(hdc, cxClient / 2, 0, NULL);
		LineTo(hdc, cxClient / 2, cyClient);
		MoveToEx(hdc, 0, cyClient / 2, NULL);
		LineTo(hdc, cxClient, cyClient / 2);
		SelectObject(hdc, hOldPen);

		// 头（直径240）
		hBlueBrush = CreateSolidBrush(RGB(0, 159, 232));
		hOldBrush = SelectObject(hdc, hBlueBrush);
		Ellipse(hdc, cxClient / 2 - 120, cyClient / 2 - 200, cxClient / 2 + 120, cyClient / 2 + 40);
		SelectObject(hdc, hOldBrush);

		// 脸（直径200）
		Ellipse(hdc, cxClient / 2 - 100, cyClient / 2 - 160, cxClient / 2 + 100, cyClient / 2 + 40);

		// 眼睛（长60， 宽50）
		Ellipse(hdc, cxClient / 2 - 50, cyClient / 2 - 180, cxClient / 2, cyClient / 2 - 120);
		Ellipse(hdc, cxClient / 2 + 50, cyClient / 2 - 180, cxClient / 2, cyClient / 2 - 120);

		hOldBrush = SelectObject(hdc, GetStockObject(BLACK_BRUSH));
		Ellipse(hdc, cxClient / 2 - 20, cyClient / 2 - 160, cxClient / 2 - 5, cyClient / 2 - 140);
		Ellipse(hdc, cxClient / 2 + 20, cyClient / 2 - 160, cxClient / 2 + 5, cyClient / 2 - 140);
		SelectObject(hdc, GetStockObject(WHITE_BRUSH));
		Ellipse(hdc, cxClient / 2 - 15, cyClient / 2 - 155, cxClient / 2 - 10, cyClient / 2 - 145);
		Ellipse(hdc, cxClient / 2 + 15, cyClient / 2 - 155, cxClient / 2 + 10, cyClient / 2 - 145);
		SelectObject(hdc, hOldBrush);

		// 鼻子
		hRedBrush = CreateSolidBrush(RGB(255, 0, 0));
		hOldBrush = SelectObject(hdc, hRedBrush);
		Ellipse(hdc, cxClient / 2 - 10, cyClient / 2 - 135, cxClient / 2 + 10, cyClient / 2 - 115);
		
		MoveToEx(hdc, cxClient / 2, cyClient / 2 - 115, NULL);
		LineTo(hdc, cxClient / 2, cyClient / 2 - 30);
		
		SelectObject(hdc, hOldBrush);

		// 嘴巴
		Arc(hdc, cxClient / 2 - 70, cyClient / 2 - 120, cxClient / 2 + 70, cyClient / 2 - 30, \
			cxClient / 2 - 60, cyClient / 2 - 50, cxClient / 2 + 60, cyClient / 2 - 50);

		// 胡子
		MoveToEx(hdc, cxClient / 2 - 70, cyClient / 2 - 115, NULL);
		LineTo(hdc, cxClient / 2 - 20, cyClient / 2 - 100);
		MoveToEx(hdc, cxClient / 2 - 80, cyClient / 2 - 85, NULL);
		LineTo(hdc, cxClient / 2 - 20, cyClient / 2 - 85);
		MoveToEx(hdc, cxClient / 2 - 70, cyClient / 2 - 55, NULL);
		LineTo(hdc, cxClient / 2 - 20, cyClient / 2 - 70);

		MoveToEx(hdc, cxClient / 2 + 70, cyClient / 2 - 115, NULL);
		LineTo(hdc, cxClient / 2 + 20, cyClient / 2 - 100);
		MoveToEx(hdc, cxClient / 2 + 80, cyClient / 2 - 85, NULL);
		LineTo(hdc, cxClient / 2 + 20, cyClient / 2 - 85);
		MoveToEx(hdc, cxClient / 2 + 70, cyClient / 2 - 55, NULL);
		LineTo(hdc, cxClient / 2 + 20, cyClient / 2 - 70);

		// 身体
		hOldBrush = SelectObject(hdc, hBlueBrush);
		Rectangle(hdc, cxClient / 2 - 90, cyClient / 2, cxClient / 2 + 90, cyClient / 2 + 150);
		SelectObject(hdc, hOldBrush);

		// 肚子
		Ellipse(hdc, cxClient / 2 - 70, cyClient / 2 - 20, cxClient / 2 + 70, cyClient / 2 + 120);
		hPen = CreatePen(PS_SOLID, 2, RGB(255, 255, 255));
		hOldPen = SelectObject(hdc, hPen);
		Arc(hdc, cxClient / 2 - 70, cyClient / 2 - 20, cxClient / 2 + 70, cyClient / 2 + 120, \
			cxClient / 2 + 60, cyClient / 2 - 10, cxClient / 2 - 60, cyClient / 2 - 10);
		SelectObject(hdc, hOldPen);

		// 项圈
		hOldBrush = SelectObject(hdc, hRedBrush);
		RoundRect(hdc, cxClient / 2 - 95, cyClient / 2 - 5, cxClient / 2 + 95, cyClient / 2 + 10, 20, 20);
		SelectObject(hdc, hOldBrush);

		// 铃铛
		hYellowBrush = CreateSolidBrush(RGB(255, 255, 0));
		hOldBrush = SelectObject(hdc, hYellowBrush);
		Ellipse(hdc, cxClient / 2 - 15, cyClient / 2, cxClient / 2 + 15, cyClient / 2 + 30);
		RoundRect(hdc, cxClient / 2 - 15, cyClient / 2 + 10, cxClient / 2 + 15, cyClient / 2 + 15, 2, 2);
		SelectObject(hdc, hRedBrush);
		Ellipse(hdc, cxClient / 2 - 4, cyClient / 2 + 18, cxClient / 2 + 4, cyClient / 2 + 26);
		MoveToEx(hdc, cxClient / 2, cyClient / 2 + 26, NULL);
		LineTo(hdc, cxClient / 2, cyClient / 2 + 30);

		SelectObject(hdc, hOldBrush);

		// 口袋
		Pie(hdc, cxClient / 2 - 50, cyClient / 2, cxClient / 2 + 50, cyClient / 2 + 100, \
			cxClient / 2 - 50, cyClient / 2 + 50, cxClient / 2 + 50, cyClient / 2 + 50);

		// 脚
		Pie(hdc, cxClient / 2 - 20, cyClient / 2 + 130, cxClient / 2 + 20, cyClient / 2 + 170, \
			cxClient / 2 + 20, cyClient / 2 + 150, cxClient / 2 - 20, cyClient / 2 + 150);
		hPen = CreatePen(PS_SOLID, 2, RGB(255, 255, 255));
		hOldPen = SelectObject(hdc, hPen);
		MoveToEx(hdc, cxClient / 2 - 20, cyClient / 2 + 150, NULL);
		LineTo(hdc, cxClient / 2 + 20, cyClient / 2 + 150);
		SelectObject(hdc, hOldPen);

		Ellipse(hdc, cxClient / 2 - 110, cyClient / 2 + 130, cxClient / 2 - 10, cyClient / 2 + 170);
		Ellipse(hdc, cxClient / 2 + 110, cyClient / 2 + 130, cxClient / 2 + 10, cyClient / 2 + 170);

		// 手
		hOldBrush = SelectObject(hdc, hBlueBrush);
		apt[0].x = cxClient / 2 - 90;
		apt[0].y = cyClient / 2 + 10;
		apt[1].x = cxClient / 2 - 130;
		apt[1].y = cyClient / 2 + 50;
		apt[2].x = cxClient / 2 - 110;
		apt[2].y = cyClient / 2 + 70;
		apt[3].x = cxClient / 2 - 90;
		apt[3].y = cyClient / 2 + 60;
		Polygon(hdc, apt, 4);
		SelectObject(hdc, hOldBrush);
		Ellipse(hdc, cxClient / 2 - 150, cyClient / 2 + 46, cxClient / 2 - 110, cyClient / 2 + 86);

		hOldBrush = SelectObject(hdc, hBlueBrush);
		apt[0].x = cxClient / 2 + 90;
		apt[0].y = cyClient / 2 + 10;
		apt[1].x = cxClient / 2 + 130;
		apt[1].y = cyClient / 2 + 50;
		apt[2].x = cxClient / 2 + 110;
		apt[2].y = cyClient / 2 + 70;
		apt[3].x = cxClient / 2 + 90;
		apt[3].y = cyClient / 2 + 60;
		Polygon(hdc, apt, 4);
		SelectObject(hdc, hOldBrush);
		Ellipse(hdc, cxClient / 2 + 150, cyClient / 2 + 46, cxClient / 2 + 110, cyClient / 2 + 86);

		hPen = CreatePen(PS_SOLID, 2, RGB(0, 159, 232));
		hOldPen = SelectObject(hdc, hPen);
		MoveToEx(hdc, cxClient / 2 - 90, cyClient / 2 + 10, NULL);
		LineTo(hdc, cxClient / 2 - 90, cyClient / 2 + 50);
		MoveToEx(hdc, cxClient / 2 + 90, cyClient / 2 + 10, NULL);
		LineTo(hdc, cxClient / 2 + 90, cyClient / 2 + 50);
		SelectObject(hdc, hOldPen);

		EndPaint(hwnd, &ps);
		return 0;

	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;
	}

	return DefWindowProc(hwnd, message, wParam, lParam);
}