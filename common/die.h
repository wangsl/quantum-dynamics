
/* $Id$ */

#ifndef DIE_H
#define DIE_H

#include <cstdlib>
#include "matutils.h"

void die_at(const char *s, const char *file, int line);
void die(const char *s);

#endif /* DIE_H */
