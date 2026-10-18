// Prints the revision and working tree state compiled into the Tempest library that Tempestpy is linked against.
// Used at configure time (see CMakeLists.txt) to check the library itself instead of only the CMake config files.
#include <iostream>

#include "storm-version-info/storm-version.h"

int main() {
    const char* state = "UNKNOWN";
    if (storm::StormVersion::dirty == storm::StormVersion::DirtyState::Clean) {
        state = "CLEAN";
    } else if (storm::StormVersion::dirty == storm::StormVersion::DirtyState::Dirty) {
        state = "DIRTY";
    }
    std::cout << storm::StormVersion::gitRevisionHash << " " << state << std::endl;
    return 0;
}
