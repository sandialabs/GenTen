add_subdirectory( src )

# Targets to install matlab source files
ADD_CUSTOM_TARGET (copy_python_dir ALL)
ADD_CUSTOM_COMMAND (
  TARGET copy_python_dir POST_BUILD
  COMMAND cmake -E copy_directory
  ${Genten_SOURCE_DIR}/python/pyGenten
  ${Genten_BINARY_DIR}/python/pyGenten
  )
# Hint:  Use trailing slash in directory name to remove it from install tree
INSTALL (
  DIRECTORY ${Genten_SOURCE_DIR}/python/pyGenten
  DESTINATION ${pyGenten_INSTALL_DIR})
