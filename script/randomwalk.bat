%windir%\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy ByPass -NoExit -Command "& 'C:\ProgramData\Anaconda3\shell\condabin\conda-hook.ps1' ; conda activate 'C:\ProgramData\Anaconda3' "
conda activate dgl
cd D:\OneDrive\SLKG
python preprocess2txt.py
python ./deepwalk/deepwalk.py --data_file graph.txt --only_gpu --gpus 0 --print_loss --window_size 5 --use_context_weight --fast_neg --num_sampler_threads 6 --num_walks 32 --walk_length 128
python emb2graph.py
