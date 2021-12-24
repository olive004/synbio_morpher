#ifndef INTARNA_CONFIG_H
#define INTARNA_CONFIG_H


/*
* The following pre-processor definitions specify whether
* or not certain features were activated upon build-time
*/

/* Name of package */
#ifndef INTARNA_PACKAGE
#define INTARNA_PACKAGE "intaRNA"
#endif

/* Define to the address where bug reports for this package should be sent. */
#ifndef INTARNA_PACKAGE_BUGREPORT
#define INTARNA_PACKAGE_BUGREPORT ""
#endif

/* Define to the full name of this package. */
#ifndef INTARNA_PACKAGE_NAME
#define INTARNA_PACKAGE_NAME "IntaRNA"
#endif

/* Define to the full name and version of this package. */
#ifndef INTARNA_PACKAGE_STRING
#define INTARNA_PACKAGE_STRING "IntaRNA 3.2.0"
#endif

/* Define to the one symbol short name of this package. */
#ifndef INTARNA_PACKAGE_TARNAME
#define INTARNA_PACKAGE_TARNAME "intaRNA"
#endif

/* Define to the home page for this package. */
#ifndef INTARNA_PACKAGE_URL
#define INTARNA_PACKAGE_URL "http://www.bioinf.uni-freiburg.de "
#endif

/* Define to the version of this package. */
#ifndef INTARNA_PACKAGE_VERSION
#define INTARNA_PACKAGE_VERSION "3.2.0"
#endif

/* Version number of package */
#ifndef INTARNA_VERSION
#define INTARNA_VERSION "3.2.0"
#endif


/* multi-threading support */
#ifndef INTARNA_MULITHREADING
#define INTARNA_MULITHREADING 1
#endif

#endif // INTARNA_CONFIG_H
