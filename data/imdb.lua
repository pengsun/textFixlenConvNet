--[[ imdb (from Rie's conText-v2.0)
--#data = n
--#words in doc = M (variable length)
Training data layout:
--instance: {n} list, each entry: {M} tensor of indices to vocabulary
--labes: {n} , 1 or 2
--Testing data layout:
-- the same
--
-- #vocabulary size = 30K
-- n = 25K 
]]--

file = require('pl.file')
stringx = require('pl.stringx')
utils = require('pl.utils')
path = require('pl.path')
require('dataProvider')

--[[ options ]]--
opt = opt or {
  dataSize = 'small',
  M = 64, -- length of the sub string
  V = 3000, -- #vocabulary size
}
local imdb_path = './data/imdb'
local fn_vocab = 'imdb_trn-30000.vocab'
local fn_trX = 'imdb-train.txt.tok'
local fn_trY = 'imdb-train.cat'
local fn_teX = 'imdb-test.txt.tok'
local fn_teY = 'imdb-test.cat'
local small = {trN = 300, teN = 100}

--[[ helper functions ]]--
local read_vocab = function (ffn)
  assert(path.isfile(ffn), ffn .. 'not exists!')
  local lines = utils.readlines(ffn)
  -- each line: word count
  local vocab = {}
  for i, line in ipairs(lines) do
    word = stringx.split(line)[1]
    vocab[word] = i
  end
  return vocab
end

local read_tokAsIndex = function (ffn, vocab)
  assert(vocab)
  assert(path.isfile(ffn), ffn .. ' not exists!')
  local lines = utils.readlines(ffn)
  -- each line: a sequence of words
  
  local ww = {}
  local missCount = 0 -- count for out-of-vocab words
  for i, line in ipairs(lines) do
    local tmp = {}
    local words = stringx.split(line)
    
    for j, word in ipairs(words) do
      word = string.lower(word)
      local ind = vocab[word]
      --if not ind then print(word .. ' out-of-voc') end
      table.insert(tmp, ind)
    end -- for word
    
    -- how many out-of-vocab words
    missCount = missCount + (#words - #tmp)
    -- convert to Tensor
    ww[i] = torch.FloatTensor(tmp)
  end -- for line'
  
  return ww, missCount
end

local read_catAsIndex = function (ffn)
  assert(path.isfile(ffn), ffn .. ' not exists!')
  local lines = utils.readlines(ffn)
  -- each line: pos (or neg)
  local cat = torch.FloatTensor(#lines)
  for i, line in ipairs(lines) do
    if line == 'pos' then
      cat[i] = 2
    elseif line == 'neg' then
      cat[i] = 1
    else
      error('unknown string ' .. cat[i] ..
        'in file ' .. ffn)
    end
  end
  return cat
end

local test = function ()
  local test_read_vocab = function ()
    local path = require('pl.path')
    local ffn = path.join(imdb_path, fn_vocab)
    --require('mobdebug').start()
    local voc = read_vocab(ffn)

    print('the' .. ' ' .. voc['the'])
    print('.' .. ' ' .. voc['.'])
    print('eustache' .. ' ' .. voc['eustache'])
    print('re-do' .. ' ' .. voc['re-do'])
  end

  local test_read_tokAsIndex = function ()
    local path = require('pl.path')

    local ffn_vocab = path.join(imdb_path, fn_vocab)
    local voc = read_vocab(ffn_vocab)

    local ffn_ww = path.join(imdb_path, fn_trX)
    --require('mobdebug').start()
    local ww, mc = read_tokAsIndex(ffn_ww, voc)

    print('#ww = ' .. #ww)
    iii = {1, 55, 8892, 10000, 20063}
    for _, i in ipairs(iii) do
      assert(ww[i]:dim() == 1)
      print(i .. ': ' .. ww[i]:numel())
    end

    print('miss count = ' .. mc)
  end

  local test_read_catAsIndex = function ()
    local path = require('pl.path')
    local ffn = path.join(imdb_path, fn_trY)
    require('mobdebug').start()
    local cat = read_catAsIndex(ffn)

    assert(cat:dim() == 1)
    print('#cat = ' .. cat:numel())
  end


  test_read_vocab()
  test_read_tokAsIndex()
  test_read_catAsIndex()
end

--[[ load training & testing raw data ]]--
-- vocabulary
if not (opt.V == 30000) then
  error('Unsupported opt.V (vocab size) ' .. opt.V)
end
local ffn_vocab = path.join(imdb_path, fn_vocab)
local vocab = read_vocab(ffn_vocab)
-- training data & testing data
local load_XY = function(fnX, fnY)
  local ffnX = path.join(imdb_path, fnX)
  local X = read_tokAsIndex(ffnX, vocab)
  local ffnY = path.join(imdb_path, fnY)
  local Y = read_catAsIndex(ffnY)
  return X, Y
end

print('[data]')
print('loading training data...')
local trX, trY = load_XY(fn_trX, fn_trY)
print('loading testing data...')
local teX, teY = load_XY(fn_teX, fn_teY)

-- subsample (if needed)
local subsample_XY = function(X, Y, n)
  assert(n >= 1)
  newX, newY = {}, torch.FloatTensor(n)
  nskip = torch.floor(#X/n)
  for i = 1, n do
    j = (i-1)*nskip + 1
    newX[i] = X[j]:clone()
    newY[i] = Y[j] -- a number
  end
  return newX, newY
end
if opt.dataSize == 'small' then
  trX, trY = subsample_XY(trX, trY, small.trN)
  teX, teY = subsample_XY(teX, teY, small.teN)
end

--[[ make training & testing data provider ]]--
local tr = dataProvider(trX, trY, opt.M)
local te = dataProvider(teX, teY, opt.M)
assert(tr.M == te.M)

print('#tr = ' .. tr:size() .. ', #te = ' .. te:size())
print('fixed length for the sub string: ' .. tr.M)
print('\n')

return tr, te