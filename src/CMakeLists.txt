#find_library(MGL mgl PATHS /usr/local/lib)

ADD_EXECUTABLE( eyeMod main.cpp findEyeCenter.cpp findEyeCenter.h findEyeCorner.cpp findEyeCorner.h constants.h helpers.cpp helpers.h utils.hpp utils.cpp ezOptionParser.hpp)
TARGET_LINK_LIBRARIES( eyeMod ${OpenCV_LIBS} )
#target_link_libraries( eyeMod ${MGL})