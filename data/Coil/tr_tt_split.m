%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    split c        %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fordname='10Train';
class_num = max(gnd);

for times=1:20
    trainIdx=[];
    testIdx = [];
    for i = 1:class_num
        mark = find(gnd==i);
        r = randperm(size(mark,1));
        trainIdx=[trainIdx' mark(r(1:10))']';
    end
    testIdx=(1:size(gnd))';
    testIdx(trainIdx)=[];
    save(['.\10Train\',int2str(times+40),'.mat'], 'trainIdx','testIdx');
end
