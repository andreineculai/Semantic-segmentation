#include <algorithm>
#include <string>
#include <vector>

#include <dirent.h>
#include <sys/types.h>

// read_directory()
//   Return an ASCII-sorted vector of filename entries in a given directory.
//   If no path is specified, the current working directory is used.
//
//   Always check the value of the global 'errno' variable after using this
//   function to see if anything went wrong. (It will be zero if all is well.)
//
std::vector <std::string> read_directory(const std::string& path = std::string())
{
	std::vector <std::string> result;
	dirent* de;
	DIR* dp;
	errno = 0;
	dp = opendir(path.empty() ? "." : path.c_str());
	if (dp)
	{
		while (true)
		{
			errno = 0;
			de = readdir(dp);
			if (de == NULL) break;
			result.push_back(std::string(de->d_name));
		}
		closedir(dp);
		std::sort(result.begin(), result.end());
	}
	return result;
}