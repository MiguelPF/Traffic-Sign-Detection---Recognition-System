function files = ListFiles(directory)

f = dir(directory);

files = [];
for i=1:size(f,1)
    if f(i).isdir==0,
        if strcmp(f(i).name(end-2:end),'ppm')==1 || strcmp(f(i).name(end-2:end),'jpg')==1 || strcmp(f(i).name(end-2:end),'png')==1,
            files = [files ; f(i)];
        end
    end
end