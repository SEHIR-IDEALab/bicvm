require 'nn'
require 'dpnn'
require 'rnn'
require 'nngraph'

local opt = {
  n_seq = 3,
  d_hid = 4,
  d_mem = 20,
  n_batch = 2,
}

local build_memory_module = function()
  local x = nn.Identity()()
  local mem_tm1 = nn.Identity()()
  local input = {x, mem_tm1}

  local x_list = nn.SplitTable(2)(x)

  -- simple scorer that produces a score s_i given {mem_tm1, x_i}
  local scorer = nn.Sequential()
      :add(nn.JoinTable(2))
      :add(nn.Linear(opt.d_hid + opt.d_mem, 1))
  -- produce a table of scores for each slice in the input
  local scores = nn.Sequential()
      :add(nn.ZipTableOneToMany())
      :add(nn.Sequencer(scorer)){mem_tm1, x_list}
  -- normalize scores via softmax
  local normalized_scores = nn.Sequential()
      :add(nn.JoinTable(2))
      :add(nn.SoftMax())
      :add(nn.SplitTable(2))(scores)
  -- expand scores for multiplcation
  local expanded_scores = nn.Sequencer(nn.Replicate(opt.d_hid, 2))(normalized_scores)

  -- multiply each slice of input by corresponding expanded scores and sum
  local attn = nn.Sequential()
      :add(nn.ZipTable())
      :add(nn.Sequencer(nn.CMulTable()))
      :add(nn.CAddTable()){expanded_scores, x_list}

  -- compute next state
  local mem = nn.Tanh()(nn.Linear(opt.d_hid, opt.d_mem)(attn))

  local output = {mem}
  return nn.gModule(input, output)
end


local x = torch.rand(opt.n_batch, opt.n_seq, opt.d_hid)
print(x)

local memory_module = build_memory_module()

memory_module = nn.Sequencer(nn.Recurrence(memory_module, opt.d_mem, 2))
print('recurrent memory module')
local inputs = {}
for t = 1, 3 do
  table.insert(inputs, x:clone())
end
print(inputs)
local out_memory = memory_module:forward(inputs)
print(out_memory)

local dout_memory = {}
for t = 1, #out_memory do
  dout_memory[t] = out_memory[t]:clone()
end
memory_module:backward(inputs, dout_memory)