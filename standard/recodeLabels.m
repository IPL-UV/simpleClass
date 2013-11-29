function Ycode = recodeLabels(Y)

nsam = size(Y,1);
classes = unique(Y);

Ycode = zeros(nsam,numel(classes)-1);   % class 0 is not encoded, we store 000000 for this class

for i=1:nsam
    pos = Y(i);
    if Y(i)~=0
        Ycode(i,pos) = 1;
    end
end
