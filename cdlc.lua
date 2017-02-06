math.randomseed(1)

require 'nn'
require 'cunn'
require 'rnn'
require 'lfs'
require 'utils'

local stringx = require('pl.stringx')


local file = require('pl.file')
local data_path = "./data/cdlc_en_tr/"

function gradUpdate(mlp, x, y, criterion, learningRate)
   local pred = mlp:forward(x)
   local err = criterion:forward(pred, y)
   local gradCriterion = criterion:backward(pred, y)
   mlp:zeroGradParameters()
   mlp:backward(x, gradCriterion)
   mlp:updateParameters(learningRate)
   return err
end

function padding(sequence,longest_seq)
  -- if extracting word embeddings! 
  new_sequence = {}
  for i = 1 , longest_seq - #sequence do
    new_sequence[i] = '<pad>'
  end
  j = 1
  for i = (longest_seq - #sequence)+1, longest_seq do
    new_sequence[i] = sequence[j]
    j = j + 1
  end
  return new_sequence
end

function map_data(data,longest_seq,L)
  local vocab = torch.load('exp_all_82_add/'..L..'_vocab')
   x = torch.Tensor(#data,longest_seq)
  for idx,item in ipairs(data) do
    all_words = stringx.split(item, sep)
    sample = torch.Tensor(longest_seq)
    all_words = padding(all_words,longest_seq)
    for k,word in ipairs(all_words) do
      if(vocab[word] ~= nil) then
        sample[k] = vocab[word]
      else
        sample[k] = 0
      end
    end
    x[idx] = sample  
  end
  return x
end


function getData(L,class)

  local pos_data = {}
  local neg_data = {}
  local longest_seq = 0
  for f in lfs.dir(data_path..L..'/'..class..'/positive') do
    local text = file.read(data_path..L..'/'..class..'/positive/'..f)
    if (text ~= nil) then
      no_words = table.getn(stringx.split(text,sep))
      if longest_seq < no_words then 
        longest_seq = no_words
      end
      table.insert(pos_data,text) 
    end
  end  
  for f in lfs.dir(data_path..L..'/'..class..'/negative') do
    local text = file.read(data_path..L..'/'..class..'/negative/'..f)
    if (text ~= nil) then
      no_words = table.getn(stringx.split(text,sep))
      if longest_seq < no_words then 
        longest_seq = no_words
      end
      table.insert(neg_data,text)
    end
  end
  local pos_mapped = map_data(pos_data,longest_seq,L)
  local neg_mapped = map_data(neg_data,longest_seq,L)
  
  return pos_mapped, neg_mapped
  
end





sep = ' '
local L1 = 'english'
local L2 = 'turkish'
local classes = {'art','arts','biology','business','creativity','culture','design','economics','education','entertainment','health','politics','science','technology'}

local model_l1 = torch.load('exp_all_82_add/'..L1..'_model'):double()
local model_l2 = torch.load('exp_all_82_add/'..L2..'_model'):double()
inp_size = 64

max_epoch = 300
lr = 0.01
lr_decay = 1
threshold = 5
folds = 10
init = 0.001
test_percentage = 0.25
train_percentage = 1
output_file = io.open("f1scorevslr_en-tr.csv", "w")

--param loop starts here
--for i = 0, -5, -1 do
--for max_epoch = 300,500,50 do
--lr = 10 ^ i
--print('lr: '..lr)
-- classes loop starts here
f1_score_avg = 0
score_avg = 0
for _,class in ipairs(classes) do
  positive, negative = getData(L1,class)
  local split = nn.SplitTable(2)
  all_raw = nn.JoinTable(1):forward{positive, negative}
  targets = nn.JoinTable(1):forward{torch.Tensor(positive:size(1)):fill(1), torch.Tensor(negative:size(1)):fill(0)}
  yy = split:forward(all_raw)
  all = model_l1:forward(split:forward(all_raw))
  
  local mlp = nn.Sequential()
  mlp:add(nn.Linear(inp_size, 1))
  mlp:add(nn.Sigmoid())
  criterion=nn.BCECriterion()
  
  
  
  

  mlp:getParameters():uniform(-1*init,init)
  score = 0
  precision_acc = 0
  recall_acc = 0
  f1_score = 0
  
  for fold = 1, folds do
    --print("Training")
    --params, gradParams = mlp:parameters()
    --all_params = torch.Tensor(max_epoch, params:size(1))
    for epoch = 1, max_epoch do
    local errors = {}
      -- loop across all the samples
      local shuffle = torch.totable(torch.randperm(all:size(1)))
      for i = 1, math.floor(all:size(1)*train_percentage) do
        x = all[shuffle[i]]
        target = targets[shuffle[i]]
        y = torch.Tensor(1):fill(target)
        err = gradUpdate(mlp, x, y, criterion, lr)
        table.insert(errors, err)
        
      end
      --all_params[epoch] = params:clone()
    --printf('epoch %4d, loss = %6.50f \n', epoch, torch.mean(torch.Tensor(errors)))
      if epoch % threshold == 0 then lr = lr / lr_decay end
    end
    
    -- averaged perceptron
    --mean = nn.Mean(1, 2)
    --params = mean:forward(all_params)
    
    
    
    
    -- get L2 data
    positive, negative = getData(L2,class)
    positive = positive[{{1,math.floor(positive:size(1)*test_percentage)},{}}]
    negative = negative[{{1,math.floor(negative:size(1)*test_percentage)},{}}]
    all_raw = nn.JoinTable(1):forward{positive, negative}
    targets = nn.JoinTable(1):forward{torch.Tensor(positive:size(1)):fill(1), torch.Tensor(negative:size(1)):fill(0)}
    all = model_l2:forward(split:forward(all_raw))
    
    
    
    
    -- test
    --print("Testing")
    correct = 0
    predicted_positives = 0
    true_positives = 0
    true_negatives = 0
    all_positives = positive:size(1)
    for i = 1, all:size(1) do
      x = all[i]
      pred = mlp:forward(x)
      if pred[1] < 0.5 then
        output = 0 
      else 
        output = 1 
        predicted_positives = predicted_positives + 1
      end
      if output == targets[i] then
        correct = correct + 1 
        if targets[i] == 1 then true_positives = true_positives + 1 end
        if targets[i] == 0 then true_negatives = true_negatives + 1 end
      end
  end
    
    if not(predicted_positives == 0 or all_positives == 0) then 
      precision = true_positives / predicted_positives
      recall = true_positives / all_positives
      precision_acc = precision_acc + precision
      recall_acc = recall_acc + recall
      f1_score = f1_score + (2 * precision * recall / (precision+recall))
      score = score + correct / all:size(1)
    else 
      fold = fold -1
    end
  
  end
  
  print("Class: "..class)
  print("Test Score: " .. (score / folds) * 100 .. "%")
  print("Precision: " .. precision_acc / folds)
  print("Recall: " .. recall_acc / folds)
  print("F1-Score: " .. f1_score / folds)
  f1_score_avg = f1_score_avg + f1_score / folds
  score_avg = score_avg +  (score / folds) * 100
end
print("Average f1-Score: " .. f1_score_avg / #classes)
print("Average Acc: " .. score_avg / #classes)
--output_file:write(lr .. ',' .. f1_score_avg / #classes .. '\n' )
--end
