require 'optim'

--[[ options ]]--
opt = opt or {
  logPath = './'
}

--[[ information, or intermediate results]]--
local classes = {'1', '2'}
local info = {tr = {}, te = {}}
for key, _ in pairs(info) do
  info[key].err = {} 
  info[key].ell = {} 
  info[key].conf = optim.ConfusionMatrix(classes)
end

--[[ text log ]]--
local curdir = paths.dirname(paths.thisfile())
local logger = {
  err = optim.Logger(paths.concat(curdir, opt.logPath, 'error.log')),
  ell = optim.Logger(paths.concat(curdir, opt.logPath, 'loss.log'))
}
logger.err:setNames{'testing error'}
logger.ell:setNames{'training loss'}

--[[ write opt, net ]]--
local write_opt_net = function (opt, md)
  --require('mobdebug').start()
  if not path.isdir(opt.logPath) then
    path.mkdir(opt.logPath)
  end
  local fn = path.join(opt.logPath, 'opt_net.txt')
  local str = 'opt = \n' .. xlua.table2string(opt, true) .. 
              '\n\n' .. md:__tostring()
  
  file.write(fn, str)
end

print('[observer]')
print('log path: ' .. paths.concat(curdir,opt.logPath))
print('\n')

return info, logger, write_opt_net