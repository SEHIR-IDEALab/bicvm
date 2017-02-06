-- In this case, an FastLSTM is nested withing a Recurrence.
require 'rnn'

-- hyper-parameters 
batchSize = 8
rho = 5 -- sequence length
hiddenSize = 7
nIndex = 10
lr = 0.1

-- Recurrence.recurrentModule
 rm = nn.Sequential()
  :add(nn.ParallelTable()
      :add(nn.LookupTable(nIndex, hiddenSize)) 
      :add(nn.Linear(hiddenSize, hiddenSize))) 
   :add(nn.CAddTable())
   :add(nn.Sigmoid())
  :add(nn.FastLSTM(hiddenSize,hiddenSize)) -- an AbstractRecurrent instance
  :add(nn.Linear(hiddenSize,hiddenSize))
  :add(nn.Sigmoid())    

 rnn = nn.Sequential()
   :add(nn.Recurrence(rm, hiddenSize, 0)) -- another AbstractRecurrent instance
   :add(nn.Linear(hiddenSize, nIndex))
   :add(nn.LogSoftMax())

-- all following code is exactly the same as the simple-sequencer-network.lua script
-- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance.
rnn = nn.Sequencer(rnn)