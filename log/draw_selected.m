function draw_selected()
%%
spec = {'ro-', 'bo-', 'co-', 'ko-', 'mo-', 'go-', 'r*--', 'b*--'};
[loss, err, names] = deal({});
%%

% fns = {...
%     'imdb_cpx2_C1024',...
%     'imdb_cpx2_C512',...
%     'imdb_cpx2_C256',...
%     'imdb_cpx2_C128',...
%     'imdb_cpdx2_C128',...
% };

fns = {...
    'imdb/cmd_C50_M128',...
    'imdb/cmd_C50_M64',...
};

tarDir = '.';
for i = 1 : numel(fns)
    
    % algo names
    names{end+1} = fns{i};
    % training loss
    loss{end+1} = get_vec( fullfile(tarDir, fns{i}, 'loss.log') );
    % testing error
    err{end+1} = get_vec( fullfile(tarDir, fns{i}, 'error.log') );
    % temp fix
    if strcmp(fns{i}, 'imdb_cpx2_C512')
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
title('testing error');
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