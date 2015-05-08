
/* $Id: timer.h 73 2013-12-07 19:24:29Z wangsl $ */

#ifndef TIMER_H
#define TIMET_H

#include <iostream>
using namespace std;
#include <unistd.h>
#include <time.h> 
#include <sys/times.h>

void print_time(const double &sec, const char *header = 0)
{
  const streamsize default_precision = cout.precision();
  
  cout.precision(2);
  
  if(header) 
    cout << " " << header;
  
  cout << " time: ";
  if(sec < 60) 
    cout << sec << " secs";
  else 
    if(sec < 3600)  
      cout << int(sec/60) << " mins, " << sec-int(sec/60)*60 << " secs";
    else 
      if(sec < 86400) 
	cout << int(sec/3600) << " hrs, " 
	     << (sec-int(sec/3600)*3600)/60.0 << " mins";
      else   
	cout << int(sec/86400) << " days, " 
	     << (sec-int(sec/86400)*86400)/3600.0 << " hrs";
  cout << endl;
  
  cout.precision(default_precision);
}

class CPUTimer
{
public:
  CPUTimer()
  { 
    reset();
    ticks_per_second = sysconf(_SC_CLK_TCK);
  }

  ~CPUTimer() { }

  void reset() { times(&start_time); }
  
  double time() const
  { 
    struct tms end_time;
    times(&end_time);
    clock_t diff = end_time.tms_utime - start_time.tms_utime;
    double seconds = ((double) diff)/ticks_per_second;
    return seconds;
  }

private:
  struct tms start_time;
  int ticks_per_second;
};

class WallTimer
{
public:
  WallTimer() { reset(); }
  
  ~WallTimer() { }
  
  void reset() { ::time(&start_time); }
  
  double time() const
  { 
    time_t end_time;
    ::time(&end_time);
    double seconds = difftime(end_time, start_time);
    return seconds;
  }

private:
  time_t start_time;
};

#endif /* TIMER_H */
