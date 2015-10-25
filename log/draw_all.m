function draw_all(tarDir)
%%
spec = {'ro-', 'bo-', 'co-', 'ko-', 'mo-', 'go-', 'r*--', 'b*--'};
[loss, err, names] = deal({});
%%
if (nargin < 1)
  tarDir = fileparts(mfilename('fullpath'));
end
fns = dir(tarDir);
isValid = @(x) (~strcmp(x.name,'.') & ...
                ~strcmp(x.name,'..') & ...
                x.isdir);
for i = 1 : numel(fns)
    if ~isValid(fns(i)), continue; end
    
    % algo names
    names{end+1} = fns(i).name;
    % training loss
    loss{end+1} = get_vec( fullfile(tarDir, fns(i).name, 'loss.log') );
    % testing error
    err{end+1} = get_vec( fullfile(tarDir, fns(i).name, 'error.log') );
    % temp fix
    if strcmp(fns(i).name, 'imdb_full_C512')
        err{end} = 1 - err{end};
    end
    
end
%% draw loss
figure;
hold on;
for i = 1 : numel(loss)
    j = mod(i-1, numel(spec)) + 1;
    plot(loss{i}, spec{j});
end
hold off;
set(gca, 'yscale','log');
grid on;
title('training loss');
legend(names, 'Interpreter','none');
%% draw testing error
figure;
hold on;
for i = 1 : numel(err)
    j = mod(i-1, numel(spec)) + 1;
    plot(err{i}, spec{j});
end
hold off;
grid on;
title('testing loss');
legend(names, 'Interpreter','none');

function v = get_vec(ffn)
v = [];
f = fopen(ffn);
while true
    str = fgetl(f);
    if ~ischar(str), break; end
    num = str2double(str);
    if isnan(num), continue; end
    v(end+1) = num; %#ok<AGROW>
end
fclose(f);
v = v(:);