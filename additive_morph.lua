require 'data_morph'
require 'nn'
require 'cunn'
require 'rnn'

L1 = 'english'
L2 = 'turkish'
data_path = "./data/train_en_tr/"

function printf(s,...)
  return io.write(s:format(...))
end

function printtime(s)
  return string.format("%.2d:%.2d:%.2d", s/(60*60), s/60%60, s%60)
end


function test_model(final,epoch)
  model_l1:evaluate()
  model_l2:evaluate()
  local input1 = inputs1[{{1,1000},{},{}}]
  local input2 = inputs2[{{1,1000},{},{}}]
  local output1 = model_l1:forward( input1)
  local output2 = model_l2:forward( input2)
  all_roots1 = output1:double()
  all_roots2 = output2:double()

  score = 0
  for idx1 = 1, all_roots1:size(1) do
    closest = idx1
    for idx2 = 1, all_roots2:size(1) do
      if torch.dist(all_roots1[idx1],all_roots2[idx2]) < torch.dist(all_roots1[idx1],all_roots2[closest]) then
        closest = idx2
      end
    end
    
    if idx1 == closest then
      score = score + 1 
    else
      if final == true then print("Closest to: "..idx1.." is: ".. closest) end
    end
  end
  print("Test Score: " .. score.. '/' .. all_roots1:size(1))
  torch.save(L2..'_model'..epoch,model_l2)
  torch.save(L2..'_LT'..epoch,lt2)
  torch.save(L1..'_model'..epoch,model_l1)
  torch.save(L1..'_LT'..epoch,lt1)
  model_l1:training()
  model_l2:training()
end

corpus_en = Corpus(L1..".10000.tok")
corpus_fr = Corpus(L2..".10000.words")
corpus_en:prepare_corpus()
corpus_fr:prepare_corpus()
no_of_sents = #corpus_en.sequences
inputs1=corpus_en:get_data():cuda()
inputs2=corpus_fr:get_data():cuda()

torch.save(L1..'_vocab',corpus_en.vocab_map)
torch.save(L2..'_vocab',corpus_fr.vocab_map)


lr = 0.2
vocab_size1 = corpus_en.no_of_morphs
vocab_size2 = corpus_fr.no_of_morphs
emb_size = 64
lr_decay = 1.1
threshold = 150
max_epoch = 2500
batch_size = 100
dump_freq = 50
init = 0.1

-- build model l1 
--model_l1 = nn.Sequential()
--local lt1 = nn.LookupTableMaskZero(vocab_size1,emb_size)
--model_l1:add( nn.Sequencer(lt1))
--model_l1:add( nn.CAddTable())
--model_l1:getParameters():uniform(-0.01,0.01)

--model_l1:cuda()

local morph_lt1 = nn.LookupTableMaskZero(vocab_size1,emb_size)

morph_lt1 = nn.Recursor(morph_lt1)

local word_mod_add1 = nn.Sequential()
word_mod_add1:add(morph_lt1)
word_mod_add1:add(nn.SplitTable(2))
word_mod_add1:add(nn.CAddTable())
--word_mod_add1:add(nn.Tanh())

word_mod_add1 = nn.Recursor(word_mod_add1)
model_l1 = nn.Sequential()
model_l1:add(nn.SplitTable(2))
model_l1:add(nn.Sequencer(word_mod_add1))
model_l1:add(nn.CAddTable())
--model_l1:add(nn.Tanh())

model_l1:cuda()

model_l2 = torch.load('turkish_model_raw_10k'):cuda()

-- build model l2

--[[local morph_lt2 = nn.LookupTableMaskZero(vocab_size2,emb_size)

morph_lt2 = nn.Recursor(morph_lt2)

local word_mod_add2 = nn.Sequential()
word_mod_add2:add(morph_lt2)
word_mod_add2:add(nn.SplitTable(2))
word_mod_add2:add(nn.CAddTable())
word_mod_add2:add(nn.Tanh())

word_mod_add2 = nn.Recursor(word_mod_add2)
model_l2 = nn.Sequential()
model_l2:add(nn.SplitTable(2))
model_l2:add(nn.Sequencer(word_mod_add2))
model_l2:add(nn.CAddTable())
model_l2:add(nn.Tanh())

model_l2:cuda()]]
--[[model_l2 = nn.Sequential()
lt2 = nn.LookupTableMaskZero(vocab_size2,emb_size)
model_l2:add( nn.SplitTable(2))
model_l2:add( nn.Sequencer(lt2))
model_l2:add( nn.CAddTable())
model_l2:add( nn.SplitTable(2))
model_l2:add( nn.SelectTable(1))
model_l2:cuda()]]

model_l1:getParameters():uniform(-1*init,init)
--model_l2:getParameters():uniform(-1*init,init)


criterion = nn.AbsCriterion()
criterion = nn.MaskZeroCriterion(criterion, 1):cuda()
beginning_time = torch.tic()
split = nn.Sequential()
split:add( nn.SplitTable(2))
split:add( nn.SelectTable(1))

model_l1:training()
model_l2:training()

for i =1,max_epoch do
    errors = {}
    local inds = torch.range(1, no_of_sents,batch_size)
    local shuffle = torch.totable(torch.randperm(inds:size(1)))
    for j = 1, no_of_sents/batch_size do 
                --get input row and target
        local start = inds[shuffle[j]]
        local endd = inds[shuffle[j]]+batch_size-1
        local input1 = inputs1[{{start,endd},{},{}}]
        local input2 = inputs2[{{start,endd},{},{}}]
        model_l1:zeroGradParameters()
        model_l2:zeroGradParameters()
        -- print( target)
        local output1 = model_l1:forward( input1)
        local output2 = split:forward(model_l2:forward( input2))
        --local output2 = model_l2:forward( input2)
        local err = criterion:forward( output1, output2)
        table.insert( errors, err)
        local gradOutputs = criterion:backward(output1, output2)
        model_l1:backward(input1, gradOutputs)
        model_l1:updateParameters(lr)
        output1 = model_l1:forward( input1)
        --output2 = split:forward(model_l2:forward( input2))
        err = criterion:forward( output2, output1)
        table.insert( errors, err)
        gradOutputs = criterion:backward(output2, output1)
        model_l2:backward(input2, gradOutputs)
        model_l2:updateParameters(lr)
    end
    printf ( 'epoch %4d, loss = %6.5f, lr = %1.4f \n', i, torch.mean( torch.Tensor( errors)),lr  )
    if i % threshold == 0 then lr = lr / lr_decay end
    if i % dump_freq == 0 then
      test_model(false,i)
    end
end

durationSeconds = torch.toc(beginning_time)
print('time elapsed:'.. printtime(durationSeconds))
test_model(true,'final')