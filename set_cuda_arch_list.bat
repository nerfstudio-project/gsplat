set "TORCH_CUDA_ARCH_LIST="
for /f "delims=" %%i in ('python get_cuda_arch_list.py') do set TORCH_CUDA_ARCH_LIST="%%i"
echo TORCH_CUDA_ARCH_LIST is set to %TORCH_CUDA_ARCH_LIST%