'''
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall
Updated by: Raeid Saqur <raeidsaqur@cs.toronto.edu>

All of the files in this directory and all subdirectories are:
Copyright (c) 2022 University of Toronto
'''

'''Concrete implementations of abstract base classes.

You don't need anything more than what's been imported here
'''

import torch
from typing import Optional, Union, Tuple, Type, Set

from a2_abcs import EncoderBase, DecoderBase, EncoderDecoderBase

# All docstrings are omitted in this file for simplicity. So please read
# a2_abcs.py carefully so that you can have a solid understanding of the
# structure of the assignment.

class Encoder(EncoderBase):

    def init_submodules(self):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.rnn, self.embedding
        # 2. You will need the following object attributes:
        #   self.source_vocab_size, self.word_embedding_size,
        #   self.pad_id, self.dropout, self.cell_type,
        #   self.hidden_state_size, self.num_hidden_layers.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules: torch.nn.{LSTM, GRU, RNN, Embedding}
        '''Initialize the parameterized submodules of this network

        This method sets the following object attributes (sets them in
        `self`):

        embedding : torch.nn.Embedding
            A layer that extracts learned token embeddings for each index in
            a token sequence. It must not learn an embedding for padded tokens.
        rnn : {torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM}
            A layer corresponding to the recurrent neural network that
            processes source word embeddings. It must be bidirectional.
        '''
        self.embedding = torch.nn.Embedding(self.source_vocab_size, self.word_embedding_size, padding_idx=self.pad_id)

        if self.cell_type == 'lstm':
            self.rnn = torch.nn.LSTM(self.word_embedding_size, self.hidden_state_size, self.num_hidden_layers,
                                     dropout=self.dropout, bidirectional=True)
        elif self.cell_type == 'gru':
            self.rnn = torch.nn.GRU(self.word_embedding_size, self.hidden_state_size, self.num_hidden_layers,
                                    dropout=self.dropout, bidirectional=True)
        elif self.cell_type == 'rnn':
            self.rnn = torch.nn.RNN(self.word_embedding_size, self.hidden_state_size, self.num_hidden_layers,
                                    dropout=self.dropout, bidirectional=True)


    def forward_pass(
            self,
            F: torch.LongTensor,
            F_lens: torch.LongTensor,
            h_pad: float = 0.) -> torch.FloatTensor:
        # Recall:
        #   F is shape (S, M)
        #   F_lens is of shape (M,)
        #   h_pad is a float
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   input seq -> |embedding| -> embedded seq -> |rnn| -> seq hidden
        # 2. You will need to use the following methods:
        #   self.get_all_rnn_inputs, self.get_all_hidden_states
        '''Defines the structure of the encoder

        Parameters
        ----------
        F : torch.LongTensor
            An integer tensor of shape ``(S, M)``, where ``S`` is the number of
            source time steps and ``M`` is the batch dimension. ``F[s, m]``
            is the token id of the ``s``-th word in the ``m``-th source
            sequence in the batch. ``F`` has been right-padded with
            ``self.pad_id`` wherever ``S`` exceeds the length of the original
            sequence.
        F_lens : torch.LongTensor
            An integer tensor of shape ``(M,)`` that stores the original
            lengths of each source sequence (and input sequence) in the batch
            before right-padding.
        h_pad : float
            The value to right-pad `h` with, wherever `x` is right-padded.

        Returns
        -------
        h : torch.FloatTensor
            A float tensor of shape ``(S, M, 2 * self.hidden_state_size)``
            where ``h[s,m,i]`` refers to the ``i``-th index of the encoder
            RNN's last layer's hidden state at time step ``s`` of the
            ``m``-th sequence in the batch. The 2 is because the forward and
            backward hidden states are concatenated. If
            ``x[s,m] == 0.``, then ``h[s,m, :] == h_pad``
        '''
        x = self.get_all_rnn_inputs(F)
        h = self.get_all_hidden_states(x, F_lens, h_pad)
        # assert False, "Fill me"
        return h

    def get_all_rnn_inputs(self, F: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   F is shape (S, M)
        #   x (output) is shape (S, M, I)
        '''Get all input vectors to the RNN at once

        Parameters
        ----------
        F : torch.LongTensor
            An integer tensor of shape ``(S, M)``, where ``S`` is the number of
            source time steps and ``M`` is the batch dimension. ``F[s, m]``
            is the token id of the ``s``-th word in the ``m``-th source
            sequence in the batch. ``F`` has been right-padded with
            ``self.pad_id`` wherever ``S`` exceeds the length of the original
            sequence.

        Returns
        -------
        x : torch.FloatTensor
            A float tensor of shape ``(S, M, I)`` of input to the encoder RNN,
            where ``I`` corresponds to the size of the per-word input vector.
            Whenever ``s`` exceeds the original length of ``F[s, m]`` (i.e.
            when ``F[s, m] == self.pad_id``), ``x[s, m, :] == 0.``
        '''
        x = self.embedding(F)
        # assert False, "Fill me"
        return x

    def get_all_hidden_states(
            self, 
            x: torch.FloatTensor,
            F_lens: torch.LongTensor,
            h_pad: float) -> torch.FloatTensor:
        # Recall:
        #   x is of shape (S, M, I)
        #   F_lens is of shape (M,)
        #   h_pad is a float
        #   h (output) is of shape (S, M, 2 * H)
        #
        # Hint:
        #   relevant pytorch modules:
        #   torch.nn.utils.rnn.{pad_packed,pack_padded}_sequence
        '''Get all encoder hidden states for from input sequences

        Parameters
        ----------
        x : torch.FloatTensor
            A float tensor of shape ``(S, M, I)`` of input to the encoder RNN,
            where ``S`` is the number of source time steps, ``M`` is the batch
            dimension, and ``I`` corresponds to the size of the per-word input
            vector. ``x[s, m, :]`` is the input vector for the ``s``-th word in
            the ``m``-th source sequence in the batch. `x` has been padded such
            that ``x[F_lens[m]:, m, :] == 0.`` for all ``m``.
        F_lens : torch.LongTensor
            An integer tensor of shape ``(M,)`` that stores the original
            lengths of each source sequence (and input sequence) in the batch
            before right-padding.
        h_pad : float
            The value to right-pad `h` with, wherever `x` is right-padded.

        Returns
        -------
        h : torch.FloatTensor
            A float tensor of shape ``(S, M, 2 * self.hidden_state_size)``
            where ``h[s,m,i]`` refers to the ``i``-th index of the encoder
            RNN's last layer's hidden state at time step ``s`` of the
            ``m``-th sequence in the batch. The 2 is because the forward and
            backward hidden states are concatenated. If
            ``x[s,m] == 0.``, then ``h[s,m, :] == h_pad``
        '''
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, F_lens, enforce_sorted=False)

        ############################## Should we be using output or h_n?
        # if self.cell_type == 'lstm':
        #     output, (h_n, c_n) = self.rnn(x_packed)
        # else:
        output_packed, h_n = self.rnn(x_packed)
        padded = torch.nn.utils.rnn.pad_packed_sequence(output_packed, padding_value=h_pad)
        h = padded[0]
        # assert False, "Fill me"
        return h


def celine_test_encoder():
    S = 10
    M = 4
    hidden_size = 6
    en = Encoder(source_vocab_size=21, hidden_state_size=7, )
    my_test_input = torch.randn(S, M, hidden_size)
    # torch.tensor([[1, 2, 3], [4, 5, 6]])
    o = en.get_all_rnn_inputs(my_test_input)
    print(f"{o.shape=}")

    result = en(my_test_input) # if it was pytorch forward
    result = en.forward_pass(my_test_input) # for csc401 assignment





class DecoderWithoutAttention(DecoderBase):
    '''A recurrent decoder without attention'''

    def init_submodules(self):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.embedding, self.cell, self.ff
        # 2. You will need the following object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}
        '''Initialize the parameterized submodules of this network

        This method sets the following object attributes (sets them in
        `self`):

        embedding : torch.nn.Embedding
            A layer that extracts learned token embeddings for each index in
            a token sequence. It must not learn an embedding for padded tokens.
        cell : {torch.nn.RNNCell, torch.nn.GRUCell, torch.nn.LSTMCell}
            A layer corresponding to the recurrent neural network that
            processes target word embeddings into hidden states. We only define
            one cell and one layer
        ff : torch.nn.Linear
            A fully-connected layer that converts the decoder hidden state
            into an un-normalized log probability distribution over target
            words
        '''
        self.embedding = torch.nn.Embedding(self.target_vocab_size, self.word_embedding_size, padding_idx=self.pad_id)

        if self.cell_type == 'lstm':
            self.cell = torch.nn.LSTMCell(self.word_embedding_size, self.hidden_state_size)
        if self.cell_type == 'gru':
            self.cell = torch.nn.GRUCell(self.word_embedding_size, self.hidden_state_size)
        if self.cell_type == 'rnn':
            self.cell = torch.nn.RNNCell(self.word_embedding_size, self.hidden_state_size)

        self.ff = torch.nn.Linear(self.hidden_state_size, self.target_vocab_size)
        # assert False, "Fill me"

    def forward_pass(
        self,
            E_tm1: torch.LongTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> Tuple[
                torch.FloatTensor, Union[
                    torch.FloatTensor,
                    Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Recall:
        #   E_tm1 is of shape (M,)
        #   htilde_tm1 is of shape (M, 2 * H)
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   logits_t (output) is of shape (M, V)
        #   htilde_t (output) is of same shape as htilde_tm1
        #
        # Hints:
        # 1. The structure of the encoder should be:
        #   encoded hidden -> |embedding| -> embedded hidden -> |rnn| ->
        #   decoded hidden -> |output layer| -> output logits
        # 2. You will need to use the following methods:
        #   self.get_current_rnn_input, self.get_current_hidden_state,
        #   self.get_current_logits
        # 3. You can assume that htilde_tm1 is not empty. I.e., the hidden state
        #   is either initialized, or t > 1.
        # 4. The output of an LSTM cell is a tuple (h, c), but a GRU cell or an
        #   RNN cell will only output h.
        '''Defines the structure of the decoder

        Parameters
        ----------
        E_tm1 : torch.LongTensor
            An integer tensor of shape ``(M,)`` denoting the target language
            token ids output from the previous decoder step. ``E_tm1[m]`` is
            the token corresponding to the ``m``-th element in the batch. If
            ``E_tm1[m] == self.pad_id``, then the target sequence has ended
        htilde_tm1 : torch.FloatTensor or tuple
            If this decoder doesn't use an LSTM cell, `htilde_tm1` is a float
            tensor of shape ``(M, self.hidden_state_size)``, where
            ``htilde_tm1[m]`` corresponds to ``m``-th element in the batch.
            If this decoder does use an LSTM cell, `htilde_tm1` is a pair of
            float tensors corresponding to the previous hidden state and the
            previous cell state.
        h : torch.FloatTensor
            A float tensor of shape ``(S, M, self.hidden_state_size)`` of
            hidden states of the encoder. ``h[s, m, i]`` is the
            ``i``-th index of the encoder RNN's last hidden state at time ``s``
            of the ``m``-th sequence in the batch. The states of the
            encoder have been right-padded such that ``h[F_lens[m]:, m]``
            should all be ignored.
        F_lens : torch.LongTensor
            An integer tensor of shape ``(M,)`` corresponding to the lengths
            of the encoded source sentences.

        Returns
        -------
        logits_t : torch.FloatTensor
            A float tensor of shape ``(M, self.target_vocab_size)``.
            ``logits_t[m]`` is an un-normalized distribution over the next
            target word for the ``m``-th sequence:
            ``Pr_b(i) = softmax(logits_t[m])``
        htilde_t : torch.FloatTensor or tuple
            Like `htilde_tm1` (either a float tensor or a pair of float
            tensors), but matching the current hidden state.
        '''
        xtilde_t = self.get_current_rnn_input(E_tm1, htilde_tm1, h, F_lens)
        htilde_t = self.get_current_hidden_state(xtilde_t, htilde_tm1)
        if self.cell_type == 'lstm':  # type(htilde_t) is tuple
            logits_t = self.get_current_logits(htilde_t[0])
        else:
            logits_t = self.get_current_logits(htilde_t)

        # assert False, "Fill me"
        return logits_t, htilde_t


    def get_first_hidden_state(
            self,
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   htilde_tm1 (output) is of shape (M, 2 * H)
        #
        # Hint:
        # 1. Ensure it is derived from encoder hidden state that has processed
        # the entire sequence in each direction. You will need to:
        # - Populate indices [0: self.hidden_state_size // 2] with the hidden
        #   states of the encoder's forward direction at the highest index in
        #   time *before padding*
        # - Populate indices [self.hidden_state_size//2:self.hidden_state_size]
        #   with the hidden states of the encoder's backward direction at time
        #   t=0
        # 2. Relevant pytorch function: torch.cat
        '''Get the initial decoder hidden state, prior to the first input

        Parameters
        ----------
        h : torch.FloatTensor
            A float tensor of shape ``(S, M, self.hidden_state_size)`` of
            hidden states of the encoder. ``h[s, m, i]`` is the
            ``i``-th index of the encoder RNN's last hidden state at time ``s``
            of the ``m``-th sequence in the batch. The states of the
            encoder have been right-padded such that
            ``h[F_lens[m]:, m]`` should all be ignored.
        F_lens : torch.LongTensor
            An integer tensor of shape ``(M,)`` corresponding to the lengths
            of the encoded source sentences.

        Returns
        -------
        htilde_0 : torch.FloatTensor
            A float tensor of shape ``(M, self.hidden_state_size)``, where
            ``htilde_0[m, i]`` is the ``i``-th index of the decoder's first
            (pre-sequence) hidden state for the ``m``-th sequence in the back

        Notes
        -----
        You will or will not need `h` and `F_lens`, depending on
        whether this decoder uses attention.

        `h` is the output of a bidirectional layer. Assume
        ``h[..., :self.hidden_state_size // 2]`` correspond to the
        hidden states in the forward direction and
        ``h[..., self.hidden_state_size // 2:]`` to those in the
        backward direction.

        In the case of an LSTM, we will initialize the cell state with zeros
        later on (don't worry about it).
        '''
        half = self.hidden_state_size // 2
        num_sentences = h.size(dim=1)
        # F_lens includes the sentence start/end tokens. Each of the M elements of the cat() call has shape [1, H].
        h_fwd = torch.cat([torch.unsqueeze(h[F_lens[m] - 1, m, 0:half], dim=0) for m in range(num_sentences)], dim=0)
        h_rev = h[0, :, half:]
        htilde_0 = torch.cat((h_fwd, h_rev), dim=1)
        ############## print("h_fwd: " + str(h_fwd.size()) + " + h_rev: " + str(h_rev.size()) + " -> htilde: " + str(htilde_0.size()))
        # assert False, "Fill me"
        return htilde_0

    def get_current_rnn_input(
            self,
            E_tm1: torch.LongTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   E_tm1 is of shape (M,)
        #   htilde_tm1 is of shape (M, 2 * H) or a tuple of two of those (LSTM)
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   xtilde_t (output) is of shape (M, Itilde)
        '''Get the current input the decoder RNN

        Parameters
        ----------
        E_tm1 : torch.LongTensor
            An integer tensor of shape ``(M,)`` denoting the target language
            token ids output from the previous decoder step. ``E_tm1[m]`` is
            the token corresponding to the ``m``-th element in the batch. If
            ``E_tm1[m] == self.pad_id``, then the target sequence has ended
        h : torch.FloatTensor
            A float tensor of shape ``(S, M, self.hidden_state_size)`` of
            hidden states of the encoder. ``h[s, m, i]`` is the
            ``i``-th index of the encoder RNN's last hidden state at time ``s``
            of the ``m``-th sequence in the batch. The states of the
            encoder have been right-padded such that ``h[F_lens[m]:, m]``
            should all be ignored.
        F_lens : torch.LongTensor
            An integer tensor of shape ``(M,)`` corresponding to the lengths
            of the encoded source sentences.

        Returns
        -------
        xtilde_t : torch.FloatTensor
            A float tensor of shape ``(M, Itilde)`` denoting the current input
            to the decoder RNN. ``xtilde_t[m, :self.word_embedding_size]``
            should be a word embedding for ``E_tm1[m]``. If
            ``E_tm1[m] == self.pad_id``, then ``xtilde_t[m] == 0.``. If this
            decoder uses attention, ``xtilde_t[m, self.word_embedding_size:]``
            corresponds to the attention context vector.

        Notes
        -----
        You will or will not need `htilde_tm1`, `h` and `F_lens`, depending on
        whether this decoder uses attention.

        ``xtilde_t[m, self.word_embedding_size:]`` should not be masked out,
        regardless of whether ``E_tm1[m] == self.pad_id``
        '''
        xtilde_t = self.embedding(E_tm1)
        # assert False, "Fill me"
        return xtilde_t

    def get_current_hidden_state(
            self,
            xtilde_t: torch.FloatTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]]) -> Union[
                    torch.FloatTensor,
                    Tuple[torch.FloatTensor, torch.FloatTensor]]:
        # Recall:
        #   xtilde_t is of shape (M, Itilde)
        #   htilde_tm1 is of shape (M, 2 * H) or a tuple of two of those (LSTM)
        #   htilde_t (output) is of same shape as htilde_tm1
        '''Calculate the decoder's current hidden state

        Converts `E_tm1` to embeddings, and feeds those embeddings into
        the recurrent cell alongside `htilde_tm1`.

        Parameters
        ----------
        xtilde_t : torch.FloatTensor
            A float tensor of shape ``(M, Itilde)`` denoting the current input
            to the decoder RNN. ``xtilde_t[m, :]`` is the input vector of the
            previous target token's embedding for batch element ``m``.
            ``xtilde_t[m, :]`` may additionally include an attention context
            vector.
        htilde_tm1 : torch.FloatTensor or tuple
            If this decoder doesn't use an LSTM cell, `htilde_tm1` is a float
            tensor of shape ``(M, self.hidden_state_size)``, where
            ``htilde_tm1[m]`` corresponds to ``m``-th element in the batch.
            If this decoder does use an LSTM cell, `htilde_tm1` is a pair of
            float tensors corresponding to the previous hidden state and the
            previous cell state.

        Returns
        -------
        htilde_t : torch.FloatTensor or tuple
            Like `htilde_tm1` (either a float tensor or a pair of float
            tensors), but matching the current hidden state.

        Notes
        -----
        This method does not account for finished target sequences. That is
        handled downstream.
        '''
        htilde_t = self.cell(xtilde_t, htilde_tm1)
        # assert False, "Fill me"
        return htilde_t

    def get_current_logits(
            self,
            htilde_t: torch.FloatTensor) -> torch.FloatTensor:
        # Recall:
        #   htilde_t is of shape (M, 2 * H), even for LSTM (cell state discarded)
        #   logits_t (output) is of shape (M, V)
        '''Calculate an un-normalized log distribution over target words

        Parameters
        ----------
        htilde_t : torch.FloatTensor
            A float tensor of shape ``(M, self.hidden_state_size)`` of the
            decoder's current hidden state (excludes the cell state in the
            case of an LSTM).

        Returns
        -------
        logits_t : torch.FloatTensor
            A float tensor of shape ``(M, self.target_vocab_size)``.
            ``logits_t[m]`` is an un-normalized distribution over the next
            target word for the ``m``-th sequence:
            ``Pr_b(i) = softmax(logits_t[m])``
        '''
        logits_t = self.ff(htilde_t)
        # assert False, "Fill me"
        return logits_t


class DecoderWithAttention(DecoderWithoutAttention):
    '''A decoder, this time with attention

    Inherits from DecoderWithoutAttention to avoid repeated code.
    '''

    def init_submodules(self):
        # Hints:
        # 1. Same as the case without attention, you must initialize the
        #   following submodules: self.embedding, self.cell, self.ff
        # 2. You will need the following object attributes:
        #   self.target_vocab_size, self.word_embedding_size, self.pad_id
        #   self.hidden_state_size, self.cell_type.
        # 3. cell_type will be one of: ['lstm', 'gru', 'rnn']
        # 4. Relevant pytorch modules:
        #   torch.nn.{Embedding, Linear, LSTMCell, RNNCell, GRUCell}
        # 5. The implementation of this function should be different from
        #   DecoderWithoutAttention.init_submodules.
        '''Initialize the parameterized submodules of this network

        This method sets the following object attributes (sets them in
        `self`):

        embedding : torch.nn.Embedding
            A layer that extracts learned token embeddings for each index in
            a token sequence. It must not learn an embedding for padded tokens.
        cell : {torch.nn.RNNCell, torch.nn.GRUCell, torch.nn.LSTMCell}
            A layer corresponding to the recurrent neural network that
            processes target word embeddings into hidden states. We only define
            one cell and one layer
        ff : torch.nn.Linear
            A fully-connected layer that converts the decoder hidden state
            into an un-normalized log probability distribution over target
            words
        '''
        self.embedding = torch.nn.Embedding(self.target_vocab_size, self.word_embedding_size, padding_idx=self.pad_id)

        if self.cell_type == 'lstm':
            self.cell = torch.nn.LSTMCell(self.word_embedding_size + self.hidden_state_size, self.hidden_state_size)
        if self.cell_type == 'gru':
            self.cell = torch.nn.GRUCell(self.word_embedding_size + self.hidden_state_size, self.hidden_state_size)
        if self.cell_type == 'rnn':
            self.cell = torch.nn.RNNCell(self.word_embedding_size + self.hidden_state_size, self.hidden_state_size)

        # ff layer also takes attention as input
        self.ff = torch.nn.Linear(self.hidden_state_size, self.target_vocab_size)

        ####################################################################################
        ####################################################################################
        ####################################################################################
        # Wait, we need to do something different here...
        ####################################################################################
        ####################################################################################
        ####################################################################################
        # assert False, "Fill me"

    def get_first_hidden_state(
            self,
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        # Hint: For this time, the hidden states should be initialized to zeros.
        '''Get the initial decoder hidden state, prior to the first input

        Parameters
        ----------
        h : torch.FloatTensor
            A float tensor of shape ``(S, M, self.hidden_state_size)`` of
            hidden states of the encoder. ``h[s, m, i]`` is the
            ``i``-th index of the encoder RNN's last hidden state at time ``s``
            of the ``m``-th sequence in the batch. The states of the
            encoder have been right-padded such that
            ``h[F_lens[m]:, m]`` should all be ignored.
        F_lens : torch.LongTensor
            An integer tensor of shape ``(M,)`` corresponding to the lengths
            of the encoded source sentences.

        Returns
        -------
        htilde_0 : torch.FloatTensor
            A float tensor of shape ``(M, self.hidden_state_size)``, where
            ``htilde_0[m, i]`` is the ``i``-th index of the decoder's first
            (pre-sequence) hidden state for the ``m``-th sequence in the back

        Notes
        -----
        You will or will not need `h` and `F_lens`, depending on
        whether this decoder uses attention.

        `h` is the output of a bidirectional layer. Assume
        ``h[..., :self.hidden_state_size // 2]`` correspond to the
        hidden states in the forward direction and
        ``h[..., self.hidden_state_size // 2:]`` to those in the
        backward direction.

        In the case of an LSTM, we will initialize the cell state with zeros
        later on (don't worry about it).
        '''
        htilde_0 = torch.zeros_like(h[0, ...])  # [M, H]
        # assert False, "Fill me"
        return htilde_0

    def get_current_rnn_input(
            self,
            E_tm1: torch.LongTensor,
            htilde_tm1: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        # Hint: Use attend() for c_t
        '''Get the current input the decoder RNN

        Parameters
        ----------
        E_tm1 : torch.LongTensor
            An integer tensor of shape ``(M,)`` denoting the target language
            token ids output from the previous decoder step. ``E_tm1[m]`` is
            the token corresponding to the ``m``-th element in the batch. If
            ``E_tm1[m] == self.pad_id``, then the target sequence has ended
        h : torch.FloatTensor
            A float tensor of shape ``(S, M, self.hidden_state_size)`` of
            hidden states of the encoder. ``h[s, m, i]`` is the
            ``i``-th index of the encoder RNN's last hidden state at time ``s``
            of the ``m``-th sequence in the batch. The states of the
            encoder have been right-padded such that ``h[F_lens[m]:, m]``
            should all be ignored.
        F_lens : torch.LongTensor
            An integer tensor of shape ``(M,)`` corresponding to the lengths
            of the encoded source sentences.

        Returns
        -------
        xtilde_t : torch.FloatTensor
            A float tensor of shape ``(M, Itilde)`` denoting the current input
            to the decoder RNN. ``xtilde_t[m, :self.word_embedding_size]``
            should be a word embedding for ``E_tm1[m]``. If
            ``E_tm1[m] == self.pad_id``, then ``xtilde_t[m] == 0.``. If this
            decoder uses attention, ``xtilde_t[m, self.word_embedding_size:]``
            corresponds to the attention context vector.

        Notes
        -----
        You will or will not need `htilde_tm1`, `h` and `F_lens`, depending on
        whether this decoder uses attention.

        ``xtilde_t[m, self.word_embedding_size:]`` should not be masked out,
        regardless of whether ``E_tm1[m] == self.pad_id``
        '''
        E_tm1_embed = self.embedding(E_tm1)  # [M, W]
        c_tm1 = self.attend(htilde_tm1, h, F_lens)  # [M, H]
        xtilde_t = torch.cat((E_tm1_embed, c_tm1), dim=1) # [M, W + H]
        # assert False, "Fill me"
        return xtilde_t

    def attend(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        '''The attention mechanism. Calculate the context vector c_t.

        Parameters
        ----------
        htilde_t : torch.FloatTensor or tuple
            Like `htilde_tm1` (either a float tensor or a pair of float
            tensors), but matching the current hidden state.
        h : torch.FloatTensor
            A float tensor of shape ``(S, M, self.hidden_state_size)`` of
            hidden states of the encoder. ``h[s, m, i]`` is the
            ``i``-th index of the encoder RNN's last hidden state at time ``s``
            of the ``m``-th sequence in the batch. The states of the
            encoder have been right-padded such that ``h[F_lens[m]:, m]``
            should all be ignored.
        F_lens : torch.LongTensor
            An integer tensor of shape ``(M,)`` corresponding to the lengths
            of the encoded source sentences.

        Returns
        -------
        c_t : torch.FloatTensor
            A float tensor of shape ``(M, self.hidden_state_size)``. The
            context vector c_t is the product of weights alpha_t and h.

        Hint: Use get_attention_weights() to calculate alpha_t.
        '''
        # [S, M]
        alpha_t = self.get_attention_weights(htilde_t, h, F_lens)
        # [M, H] = [S, M, H] * [S, M]
        M = h.size(dim=1)
        H = h.size(dim=2)
        ##################### print("S: " + str(alpha_t.size(dim=0)) + " M: " + str(M) + " H: " + str(H))
        ##################### print("alpha_t: " + str(alpha_t.size()) + " h: " + str(h.size()))
        c_t = torch.zeros_like(h[0, ...])  # (M, H))
        # for m in range(M):
        #     c_t[m] = [H]
        #     # h[:, m, :] : [S, H]
        #     # alpha_t[s, :] : [1, S]
        #     for s in S:
        #         # [H] * scalar
        #         for i in I:
        #             c_t[m] += h[s, m, :] * alpha_t[s, m]
        # # c_t [M, H]
        for m in range(M):
            ########## print("_alpha_t: " + str(torch.unsqueeze(alpha_t[:, m], dim=0).size()) + " _h: " + str(h[:, m, :].size()))
            c_t[m, :] = torch.unsqueeze(alpha_t[:, m], dim=0) @ h[:, m, :]
            # c_t[m, :] = torch.mul(torch.t(h[:, m, :]), alpha_t[:, m])
        # assert False, "Fill me"
        return c_t

    def get_attention_weights(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        # DO NOT MODIFY! Calculates attention weights, ensuring padded terms
        # in h have weight 0 and no gradient. You have to implement
        # get_attention_scores()
        # alpha_t (output) is of shape (S, M)
        e_t = self.get_attention_scores(htilde_t, h)
        pad_mask = torch.arange(h.shape[0], device=h.device)
        pad_mask = pad_mask.unsqueeze(-1) >= F_lens.to(h.device)  # (S, M)
        e_t = e_t.masked_fill(pad_mask, -float('inf'))
        return torch.nn.functional.softmax(e_t, 0)

    def get_attention_scores(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor) -> torch.FloatTensor:
        # Recall:
        #   htilde_t is of shape (M, 2 * H)
        #   h is of shape (S, M, 2 * H)
        #   e_t (output) is of shape (S, M)
        #
        # Hint:
        # Relevant pytorch function: torch.nn.functional.cosine_similarity
        if type(htilde_t) is tuple:
            e_t = torch.nn.functional.cosine_similarity(h, htilde_t[0].expand_as(h), dim=2)
        else:
            e_t = torch.nn.functional.cosine_similarity(h, htilde_t.expand_as(h), dim=2)
        # assert False, "Fill me"
        return e_t

class DecoderWithMultiHeadAttention(DecoderWithAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.W is not None, 'initialize W!'
        assert self.Wtilde is not None, 'initialize Wtilde!'
        assert self.Q is not None, 'initialize Q!'

    def init_submodules(self):
        super().init_submodules()  # Do not change this line

        # Hints:
        # 1. The above line should ensure self.ff, self.embedding, self.cell are
        #    initialized
        # 2. You need to initialize the following submodules:
        #       self.W, self.Wtilde, self.Q
        # 3. You will need the following object attributes:
        #       self.hidden_state_size
        # 4. self.W, self.Wtilde, and self.Q should process all heads at once. They
        #    should not be lists!
        # 5. You do *NOT* need self.heads at this point
        # 6. Relevant pytorch module: torch.nn.Linear (note: set bias=False!)
        self.W = torch.nn.Linear(self.hidden_state_size, self.hidden_state_size, bias=False)
        self.Wtilde = torch.nn.Linear(self.hidden_state_size, self.hidden_state_size, bias=False)
        self.Q = torch.nn.Linear(self.hidden_state_size, self.hidden_state_size, bias=False)
        # assert False, "Fill me"

    def attend(
            self,
            htilde_t: Union[
                torch.FloatTensor,
                Tuple[torch.FloatTensor, torch.FloatTensor]],
            h: torch.FloatTensor,
            F_lens: torch.LongTensor) -> torch.FloatTensor:
        # Hints:
        # 1. You can use super().attend to call for the regular attention
        #   function.
        # 2. Relevant pytorch function:
        #   tensor().view, tensor().repeat_interleave
        # 3. Fun fact:
        #   tensor([1,2,3,4]).repeat(2) will output tensor([1,2,3,4,1,2,3,4]).
        #   tensor([1,2,3,4]).repeat_interleave(2) will output
        #   tensor([1,1,2,2,3,3,4,4]), just like numpy.repeat.
        # 4. You *WILL* need self.heads at this point
        '''
        Parameters
        ----------
        htilde_t : torch.FloatTensor or tuple
            Like `htilde_tm1` (either a float tensor or a pair of float
            tensors), but matching the current hidden state.
        h : torch.FloatTensor
            A float tensor of shape ``(S, M, self.hidden_state_size)`` of
            hidden states of the encoder. ``h[s, m, i]`` is the
            ``i``-th index of the encoder RNN's last hidden state at time ``s``
            of the ``m``-th sequence in the batch. The states of the
            encoder have been right-padded such that ``h[F_lens[m]:, m]``
            should all be ignored.
        F_lens : torch.LongTensor
            An integer tensor of shape ``(M,)`` corresponding to the lengths
            of the encoded source sentences.

        Returns
        -------
        c_t : torch.FloatTensor
            A float tensor of shape ``(M, self.hidden_state_size)``. The
            context vector c_t is the product of weights alpha_t and h.
        '''

        # "Key" -> Encoder -> W
        # "Value" -> Decoder -> Wtilde

        # super().attend() takes 3 arguments, with the following shapes:
        #
        # htilde_t: (M*n, hidden_state_size/n) (assuming simple rnn cell_type)
        # h: (S, M*n, hidden_state_size/n)
        # F_lens: (M*n,)

        # M: Samples
        # S: Seq. len.
        # H: Embedding size

        htilde_mod = self.Wtilde(htilde_t) if type(htilde_t) is not tuple else self.Wtilde(htilde_t[0])
        h_mod = self.W(h)
        ################# print(h.size())
        S, M, H = h.size()
        htilde_split = htilde_mod.view(M * self.heads, H // self.heads)
        h_split = h_mod.view(S, M * self.heads, H // self.heads)
        F_lens_split = F_lens.repeat_interleave(self.heads)

        c_t_split = super().attend(htilde_split, h_split, F_lens_split)
        c_t = c_t_split.view(M, H)
        c = self.Q(c_t)
        # # Split n heads.
        # partition_size = self.hidden_state_size // self.heads
        # # H * H -> H * (H/n)
        # W_split = torch.split(self.W.weight, partition_size)
        # Wtilde_split = torch.split(self.Wtilde.weight, partition_size)
        # h_split = torch.split(h)
        # htilde_split = []
        # for W, Wtilde in zip(W_split, Wtilde_split):
        #     h_split.append(W * h)
        #     htilde_split.append(Wtilde * htilde_t)
        #
        # h_split = torch.split(self.W(h), self.hidden_state_size // self.heads)
        # htilde_split = torch.split(self.Wtilde(htilde_t), self.hidden_state_size // self.heads)
        # for m in range(h.size(dim=1)):
        #
        #
        # c_t_list = []
        # for i in range(self.heads):
        #     c_t_list = super().attend(super.attend(htilde_split[i, :], h_split[i, :], F_lens))
        # c_t = torch.stack(c_t_list, dim=0)
        # c = self.Q * c_t

        # assert False, "Fill me"
        return c

class EncoderDecoder(EncoderDecoderBase):

    def init_submodules(
            self,
            encoder_class: Type[EncoderBase],
            decoder_class: Type[DecoderBase]):
        # Hints:
        # 1. You must initialize the following submodules:
        #   self.encoder, self.decoder
        # 2. encoder_class and decoder_class inherit from EncoderBase and
        #   DecoderBase, respectively.
        # 3. You will need the following object attributes:
        #   self.source_vocab_size, self.source_pad_id,
        #   self.word_embedding_size, self.encoder_num_hidden_layers,
        #   self.encoder_hidden_size, self.encoder_dropout, self.cell_type,
        #   self.target_vocab_size, self.target_eos, self.heads
        # 4. Recall that self.target_eos doubles as the decoder pad id since we
        #   never need an embedding for it

        #################################################################
        #################################################################
        #################################################################
        #################################################################
        #################################################################
        #################################################################
        #################################################################
        print("CUDA available: " + str(torch.cuda.is_available()))
        if torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            print("Current device: " + str(torch.cuda.device(device_idx)))
            print("Num devices: " + str(torch.cuda.device_count()))
            print("Device name: " + torch.cuda.get_device_name(device_idx))
        #################################################################
        #################################################################
        #################################################################
        #################################################################
        #################################################################
        #################################################################
        #################################################################


        self.encoder = encoder_class(
            self.source_vocab_size,
            self.source_pad_id,
            self.word_embedding_size,
            self.encoder_num_hidden_layers,
            self.encoder_hidden_size,
            self.encoder_dropout,
            self.cell_type)
        self.decoder = decoder_class(
            self.target_vocab_size,
            self.target_eos,  # decoder pad_id
            self.word_embedding_size,
            self.encoder_hidden_size * 2,
            self.cell_type,
            self.heads)
        # assert False, "Fill me"

    def get_logits_for_teacher_forcing(
            self,
            h: torch.FloatTensor,
            F_lens: torch.LongTensor,
            E: torch.LongTensor) -> torch.FloatTensor:
        # Recall:
        #   h is of shape (S, M, 2 * H)
        #   F_lens is of shape (M,)
        #   E is of shape (T, M)
        #   logits (output) is of shape (T - 1, M, Vo)
        #
        # Hints:
        # 1. Relevant pytorch modules: torch.{zero_like, stack}
        # 2. Recall an LSTM's cell state is always initialized to zero.
        # 3. Note logits sequence dimension is one shorter than E (why?)
        '''Get un-normed distributions over next tokens via teacher forcing

        Parameters
        ----------
        h : torch.FloatTensor
            A float tensor of shape ``(S, M, 2 * self.encoder_hidden_size)`` of
            hidden states of the encoder. ``h[s, m, i]`` is the
            ``i``-th index of the encoder RNN's last hidden state at time ``s``
            of the ``m``-th sequence in the batch. The states of the
            encoder have been right-padded such that ``h[F_lens[m]:, m]``
            should all be ignored.
        F_lens : torch.LongTensor
            An integer tensor of shape ``(M,)`` corresponding to the lengths
            of the encoded source sentences.
        E : torch.LongTensor
            A long tensor of shape ``(T, M)`` where ``E[t, m]`` is the
            ``t-1``-th token in the ``m``-th target sequence in the batch.
            ``E[0, :]`` has been populated with ``self.target_sos``. Each
            sequence has had at least one ``self.target_eos`` token appended
            to it. Further EOS right-pad the shorter sequences to make up the
            length.

        Returns
        -------
        logits : torch.FloatTensor
            A float tensor of shape ``(T - 1, M, self.target_vocab_size)``
            where ``logits[t, m, :]`` is the un-normalized log-probability
            distribution predicting the ``t``-th token of the ``m``-th target
            sequence in the batch.

        Notes
        -----
        You need not worry about handling padded values of `E` here - it will
        be handled in the loss function.
        '''

        # logits = torch.zeros_like(E)

        # for m in range(1, E.size(dim=1)):
        #     for t in range(E.size(dim=0)):
        #         v = E[t, m]
        #         logits[t, m, v] = 1.0
        htilde_t = None
        logits_list = []
        # logits = torch.empty((0, F_lens.size(dim=0), self.target_vocab_size))
        for t in range(1, E.size(dim=0)):
            logits_t, htilde_t = self.decoder(E[t-1, :], htilde_t, h, F_lens)
            logits_list.append(logits_t)
            # logits_t has shape [M, V]; unsqueeze to [1, M, V].
            # logits_t = torch.unsqueeze(logits_t, dim=0)
            # if logits is None:
            #     logits = logits_t
            # else:
            # = torch.stack((logits, logits_t))
        # assert False, "Fill me"
        logits = torch.stack(logits_list, dim=0)
        return logits

    def update_beam(
            self,
            htilde_t: torch.FloatTensor,
            b_tm1_1: torch.LongTensor,
            logpb_tm1: torch.FloatTensor,
            logpy_t: torch.FloatTensor) -> Tuple[
                torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        # perform the operations within the psuedo-code's loop in the
        # assignment.
        # You do not need to worry about which paths have finished, but DO NOT
        # re-normalize logpy_t.
        #
        # Recall
        #   htilde_t is of shape (M, K, 2 * H) or a tuple of two of those (LSTM)
        #   logpb_tm1 is of shape (M, K)
        #   b_tm1_1 is of shape (t, M, K)
        #   b_t_0 (first output) is of shape (M, K, 2 * H) or a tuple of two of
        #      those (LSTM)
        #   b_t_1 (second output) is of shape (t + 1, M, K)
        #   logpb_t (third output) is of shape (M, K)
        #
        # Hints:
        # 1. Relevant pytorch modules:
        #   torch.{flatten, topk, unsqueeze, expand_as, gather, cat}
        # 2. If you flatten a two-dimensional array of shape z of (A, B),
        #   then the element z[a, b] maps to z'[a*B + b]
        '''Update the beam in a beam search for the current time step

        Parameters
        ----------
        htilde_t : torch.FloatTensor
            A float tensor of shape
            ``(M, self.beam_with, 2 * self.encoder_hidden_size)`` where
            ``htilde_t[m, k, :]`` is the hidden state vector of the ``k``-th
            path in the beam search for batch element ``m`` for the current
            time step. ``htilde_t[m, k, :]`` was used to calculate
            ``logpy_t[m, k, :]``.
        b_tm1_1 : torch.LongTensor
            A long tensor of shape ``(t, M, self.beam_width)`` where
            ``b_tm1_1[t', m, k]`` is the ``t'``-th target token of the
            ``k``-th path of the search for the ``m``-th element in the batch
            up to the previous time step (including the start-of-sequence).
        logpb_tm1 : torch.FloatTensor
            A float tensor of shape ``(M, self.beam_width)`` where
            ``logpb_tm1[m, k]`` is the log-probability of the ``k``-th path
            of the search for the ``m``-th element in the batch up to the
            previous time step. Log-probabilities are sorted such that
            ``logpb_tm1[m, k] >= logpb_tm1[m, k']`` when ``k <= k'``.
        logpy_t : torch.FloatTensor
            A float tensor of shape
            ``(M, self.beam_width, self.target_vocab_size)`` where
            ``logpy_t[m, k, v]`` is the (normalized) conditional
            log-probability of the word ``v`` extending the ``k``-th path in
            the beam search for batch element ``m``. `logpy_t` has been
            modified to account for finished paths (i.e. if ``(m, k)``
            indexes a finished path,
            ``logpy_t[m, k, v] = 0. if v == self.eos else -inf``)

        Returns
        -------
        b_t_0, b_t_1, logpb_t : torch.FloatTensor, torch.LongTensor
            `b_t_0` is a float tensor of shape ``(M, self.beam_width,
            2 * self.encoder_hidden_size)`` of the hidden states of the
            remaining paths after the update. `b_t_1` is a long tensor of shape
            ``(t + 1, M, self.beam_width)`` which provides the token sequences
            of the remaining paths after the update. `logpb_t` is a float
            tensor of the same shape as `logpb_t`, indicating the
            log-probabilities of the remaining paths in the beam after the
            update. Paths within a beam are ordered in decreasing log
            probability:
            ``logpb_t[m, k] >= logpb_t[m, k']`` implies ``k <= k'``

        Notes
        -----
        While ``logpb_tm1[m, k]``, ``htilde_t[m, k]``, and ``b_tm1_1[:, m, k]``
        refer to the same path within a beam and so do ``logpb_t[m, k]``,
        ``b_t_0[m, k]``, and ``b_t_1[:, m, k]``,
        it is not necessarily the case that ``logpb_tm1[m, k]`` extends the
        path ``logpb_t[m, k]`` (nor ``b_t_1[:, m, k]`` the path
        ``b_tm1_1[:, m, k]``). This is because candidate paths are re-ranked in
        the update by log-probability. It may be the case that all extensions
        to ``logpb_tm1[m, k]`` are pruned in the update.

        ``b_t_0`` extracts the hidden states from ``htilde_t`` that remain
        after the update.
        '''
        # htilde_t: The hidden states in the previous beam (size b)
        # b_tm1_1: The partial results (sequence of target tokens) of the previous hidden states (size b * t)
        # logpb_tm1: Cumulative probability of the previous beam (size b)
        # logpy_t: Probabilities of continuation words for each beam (size b * V)
        #
        # We basically want to multiply logpb_tm1 and logpy_t -> P(B_k) * P(v|B_k) for all v in V, k in beams,
        # take the top k results, and put them into the next beam.
        if self.cell_type == 'lstm':
            htilde = htilde_t[0]
            cell = htilde_t[1]
        else:
            htilde = htilde_t
        # top_k_values, top_k_indices have shape [M, B, K] where B = K = self.beam_width.
        # B represents the beam index in previous iteration.
        # K represents the descending-sorted top-k continuations of that beam.
        top_k_values, top_k_indices = torch.topk(logpy_t, k=self.beam_width, dim=2)
        b_t_0 = torch.zeros_like(htilde)
        if self.cell_type == 'lstm':
            b_t_0_cell = torch.zeros_like(htilde)
        b_t_1 = torch.zeros_like(b_tm1_1)
        # Expand dim=0 by 1
        b_t_1 = torch.cat((b_t_1, torch.zeros_like(b_t_1[0, ...].unsqueeze(dim=0))), dim=0)
        logpb_t = torch.zeros_like(logpb_tm1)
        for m in range(htilde.size(dim=0)):
            candidates = []
            for b in range(self.beam_width):
                # True probabilities of the top k continuations from beam b. Using addition as we have log probs
                top_k_next = top_k_values[m, b, :] + logpb_tm1[m, b]
                for k in range(self.beam_width):
                    logpb = top_k_next[k]
                    v = top_k_indices[m, b, k]
                    candidates.append([logpb, b, v])
            candidates.sort(key=lambda x: x[0], reverse=True)
            # Take the top k candidates
            for i in range(self.beam_width):
                logpb, b, v = candidates[i]
                # m: Batch index
                # b: (Prev) beam index
                # i: ith of top k results -> Current beam index
                # v: Target word index
                # value: Log-prob of beam b + word v
                b_t_0[m, i, :] = htilde[m, b, :]
                if self.cell_type == 'lstm':
                    b_t_0_cell[m, i, :] = cell[m, b, :]
                b_t_1[:-1, m, i] = b_tm1_1[:, m, b]
                b_t_1[-1, m, i] = v
                logpb_t[m, i] = logpb

        if self.cell_type == 'lstm':
            b_t_0 = [b_t_0, b_t_0_cell]
        # assert False, "Fill me"
        return b_t_0, b_t_1, logpb_t