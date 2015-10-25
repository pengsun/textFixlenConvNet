--[[ (Conv + Max)*2 + Concatenate + Dropout]]--

require 'nn'

opt = opt or {
  V = 30000,
  C = 128,
}

--[[ model ]]--
local V = opt.V
local C = opt.C
local HU = 1000 -- #hidden units
local function make_ConvMax(winSz)
  local mo = nn.Sequential()
  mo:add(nn.TemporalConvolution(C, HU, winSz))
  mo:add(nn.ReLU(true))
  mo:add(nn.Max(2))
  return mo
end
local function make_ConvMaxCat()
  local con = nn.Concat(2) -- at dim 2 the channel
  con:add(make_ConvMax(2))
  con:add(make_ConvMax(3))
  return con
end
local md = nn.Sequential()
-- word2vec Layer
md:add(nn.LookupTable(V, C)) -- (vocabulary size, #channeds)
-- concatenate two Conv + Max layer
md:add(make_ConvMaxCat())
md:add(nn.Dropout(0.5))
-- Output Layer
md:add(nn.Linear(2*HU, 2)) -- binary classification
md:add(nn.LogSoftMax())
md:float()

--[[ weight re-initialization ]]--
local reinit
function reinit (mo)
  local std = 0.05
  for k, v in ipairs(mo.modules) do
    -- parameter layer
    if v.weight then v.weight:normal(0, std) end
    if v.bias then v.bias:zero() end
    -- look into container
    if torch.type(v) == 'nn.Concat' then
      for _, vv in ipairs(v) do reinit(v) end
    end
  end
end
reinit(md)

--[[ loss ]]--
local loss = nn.ClassNLLCriterion()
--local loss = nn.CrossEntropyCriterion()
loss:float()

--[[ manipulators ]]--
local print_size = function()
  print('model data size flow:')
  -- Modules
  local tmpl = '(%d): %s %s'
  for i = 1, #md.modules do
    local str = string.format(tmpl, i, 
      md.modules[i]:__tostring(),
      md.modules[i].output:size():__tostring__())
    print(str)
  end
  print('\n')
end

print('[net]')
print(md)
print('\n')

return md, loss, print_size