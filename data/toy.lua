--[[ toy data: randomly generated
--#data = n
--#words in doc = M (variable length)
data layout
--instance: {n} list, each entry: {M} tensor of indices to vocabulary
--labes: {n}, 1 or 2
]]--

require('torch')
require('dataProvider')

--[[ options ]]--
opt = opt or {
  dataSize = 'small',
  V = 3000, -- #vocabulary size
}

--[[ training & testing raw data ]]--
local trN, teN = 25000, 25000 -- #instances
local V = opt.V 
local Mmin, Mmax = 80, 120 -- #words in a doc

local gen_rand_data = function(n)
  -- instances
  local X = {}
  for i = 1, n do
    -- #words for the i-th doc
    local M = torch.floor( torch.uniform(Mmin, Mmax) ) 
    local tmp = torch.rand(M, 'torch.FloatTensor')
    X[i] = torch.ceil( tmp:mul(V) )
  end
  -- labels
  local tmp = torch.rand(n, 'torch.FloatTensor')
  local Y = torch.ceil( tmp:mul(2) ) -- binary 1 or 2 labels
  Y = Y:float()
  return X, Y
end

if opt.dataSize == 'small' then
  trN, teN = 500, 30
end

local trX, trY = gen_rand_data(trN)
local teX, teY = gen_rand_data(teN)

--[[ training & testing data generator ]]--
local tr, te = dataProvider(trX, trY), dataProvider(teX, teY)

print('[data]')
print('#tr = ' .. tr:size() .. ', #te = ' .. te:size())
print('\n')

return tr, te