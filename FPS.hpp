#pragma once

#include <windows.h>

class Interval {
private:
	unsigned int initial_;

public:
	// Ctor
	inline Interval() : initial_(GetTickCount()) {}

	// Dtor
	virtual ~Interval() {}

	inline unsigned int value() const {
		return GetTickCount() - initial_;
	}
};

class FPS {

protected:
	unsigned int m_fps;
	unsigned int m_fpscount;
	Interval m_fpsinterval;

public:
	FPS();
	~FPS();

	void update();
	unsigned int get() const;
	void showInConsole(bool flush = false);
};

