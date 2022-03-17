######################################
# personal properties for -D Option
######################################

set(PROBABILITY "DOUBLE_PROB")

if(LONG_PROBABILITY)
  set(PROBABILITY "LONGDOUBLE_PROB")
ENDIF(LONG_PROBABILITY)

if(QUAD_PROBABILITY)
  set(PROBABILITY "QUAD_PROB")
ENDIF(QUAD_PROBABILITY)

if(LONG_COSTS)
  SET(COST LONGLONG_COST)
ELSE(LONG_COSTS)
  set(COST INT_COST)
ENDIF(LONG_COSTS)

IF(WIN32)
  #add_definitions(-DWIN32 -D_DEBUG -D_WINDOWS -D_USRDLL )
  #add_definitions(-D__MINGW32__)

  IF(TOULBAR2)
    set_property(
      TARGET toulbar2${EXE}
      PROPERTY COMPILE_DEFINITIONS WCSPFORMATONLY INT_COST DOUBLE_PROB WIN32 _DEBUG _WINDOWS WINDOWS)
  ENDIF(TOULBAR2)
  
  IF(MENDELSOFT)
    set_property(
      TARGET mendelsoft${EXE}
      PROPERTY COMPILE_DEFINITIONS WCSPFORMATONLY INT_COST DOUBLE_PROB MENDELSOFT WIN32 _DEBUG _WINDOWS WINDOWS)
  ENDIF(MENDELSOFT)

  IF(LIBTB2)
    set_property(
      TARGET tb2
      PROPERTY COMPILE_DEFINITIONS INT_COST DOUBLE_PROB WIN32 _DEBUG _WINDOWS WINDOWS)
  ENDIF(LIBTB2)
  
ELSE(WIN32)
  
  IF(TOULBAR2)
    set_property(
      TARGET toulbar2
      PROPERTY COMPILE_DEFINITIONS WCSPFORMATONLY ${COST} ${XMLFLAG} LINUX ${boostflag} ${mpiflag} ${PROBABILITY})
    set_property(
      TARGET tb2-archive
      PROPERTY COMPILE_DEFINITIONS ${COST} ${XMLFLAG} LINUX ${boostflag} ${mpiflag} ${PROBABILITY})
    set_property(
      TARGET tb2-objects
      PROPERTY COMPILE_DEFINITIONS ${COST} ${XMLFLAG} LINUX ${boostflag} ${mpiflag} ${PROBABILITY})
  ENDIF(TOULBAR2)
  
  IF(MENDELSOFT)
    set_property(
      TARGET mendelsoft
      PROPERTY COMPILE_DEFINITIONS WCSPFORMATONLY ${COST} MENDELSOFT LINUX ${boostflag} ${mpiflag} ${PROBABILITY})
  ENDIF(MENDELSOFT)
  
  IF(LIBTB2)
    set_property(
      TARGET tb2-PIC-objects
      PROPERTY COMPILE_DEFINITIONS ${COST} ${XMLFLAG} LINUX ${boostflag} ${mpiflag} ${PROBABILITY})
    set_property(
      TARGET tb2
      PROPERTY COMPILE_DEFINITIONS ${COST} ${XMLFLAG} LINUX ${boostflag} ${mpiflag} ${PROBABILITY})
    IF(PYTB2)
      set_property(
	TARGET pytb2
	PROPERTY COMPILE_DEFINITIONS ${COST} ${XMLFLAG} LINUX ${boostflag} ${mpiflag} ${PROBABILITY})
    ENDIF(PYTB2)
  ENDIF(LIBTB2)
  
  if(ILOG)
    set_property(
      TARGET iloglue
      PROPERTY COMPILE_DEFINITIONS WCSPFORMATONLY INT_COST ILOGLUE IL_STD LINUX ${PROBABILITY})
  ENDIF(ILOG)
  
ENDIF(WIN32)