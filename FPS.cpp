#pragma once

#include <iostream>
#include "FPS.hpp"


FPS::FPS() : m_fps(0), m_fpscount(0) {

}

void FPS::update() {
	// increase the counter by one
	m_fpscount++;

	// one second elapsed? (= 1000 milliseconds)
	if (m_fpsinterval.value() > 1000)
	{
		// save the current counter value to m_fps
		m_fps = m_fpscount;

		// reset the counter and the interval
		m_fpscount = 0;
		m_fpsinterval = Interval();
	}
}

unsigned int FPS::get() const {
	return m_fps;
}

void FPS::showInConsole(bool flush) {
	std::cout << '\r' << "[INFO] FPS: " << this->get();
	if (flush) std::cout << std::flush;
}

FPS::~FPS() {}
