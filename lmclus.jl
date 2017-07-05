using LMCLUS

include("NMI.jl")

# call : genPredLab(Ms, length(trueLab))
function genPredLab(Ms, l)
    predLab = ones(l);

    for i=1:length(Ms)
        lab = labels(Ms[i]);
        predLab[lab] = i;
    end

    return predLab;
end

# Normalizes (min-max) an array
# each column is taken as a record  
function normalizeArr(x)
    for i=1:size(x)[1]
        x[i,:] = (x[i,:]-minimum(x[i,:]))/(maximum(x[i,:])-minimum(x[i,:]));
    end
end

# function to print the confusion matrix
function printCArr(c)
    for i = 1:size(c)[1]
        print("\n");
        for j = 1:size(c)[2]
            @printf "%6d" c[i,j];
        end
        print("\n");
    end
end

# x = readdlm("D:/data_backup/dataC10WL.csv", header = false, ',')';
x = readdlm("D:/Datasets/dataWL5dpert.csv", header = false, ',')';

trueLab = x[end, :];
x = x[1:(end-1), :];

# trueLab2 = x2[end, :];
# x2 = x2[1:(end-1), :];

# Normalize
normalizeArr(x);

params = LMCLUSParameters(size(x)[1] - 1);

rArr = zeros(1,3);

#tic();

for a=1:3
    tic();
    print("\n");
    print(a);

    Ms = lmclus(x, params);

    predLab = genPredLab(Ms, length(trueLab));

    n = max(length(Ms), length(trueLab))

    c = zeros(n, n);

    for i = 1:length(predLab)
        c[Int(predLab[i]), Int(trueLab[i])] += 1;
    end

    # printCArr(c);
    rArr[a] = getNMI(c);
    toc();    
end

#toc();

print(mean(rArr));