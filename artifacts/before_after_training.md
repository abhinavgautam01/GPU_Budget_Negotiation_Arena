# Before / After Training Transcript

Same task and seed, before training versus the trained lightweight selector. This is qualitative evidence that the reward signal changes behavior, not just a score table.

## Before Training

- Policy: `before_random_validish`
- Task: `coalition_market`
- Seed: `5`
- Episode reward: `0.1814`

### Step 1

- Action: `{"action_type": "reject_offer", "offer_id": "o_1"}`
- Result: `{"code": "offer_rejected", "message": "Rejected offer o_1.", "ok": true}`
- Reward: `0.008`

### Step 2

- Action: `{"action_type": "reject_offer", "offer_id": "o_2"}`
- Result: `{"code": "offer_rejected", "message": "Rejected offer o_2.", "ok": true}`
- Reward: `0.008`

### Step 3

- Action: `{"action_type": "reject_offer", "offer_id": "o_3"}`
- Result: `{"code": "offer_rejected", "message": "Rejected offer o_3.", "ok": true}`
- Reward: `0.008`

### Step 4

- Action: `{"action_type": "reject_offer", "offer_id": "o_6"}`
- Result: `{"code": "offer_rejected", "message": "Rejected offer o_6.", "ok": true}`
- Reward: `0.008`

### Step 5

- Action: `{"action_type": "reject_offer", "offer_id": "o_7"}`
- Result: `{"code": "offer_rejected", "message": "Rejected offer o_7.", "ok": true}`
- Reward: `0.008`

### Step 6

- Action: `{"action_type": "wait"}`
- Result: `{"code": "waited", "message": "No action taken.", "ok": true}`
- Reward: `0.0`

### Step 7

- Action: `{"action_type": "wait"}`
- Result: `{"code": "waited", "message": "No action taken.", "ok": true}`
- Reward: `0.0`

### Step 8

- Action: `{"action_type": "reject_offer", "offer_id": "o_8"}`
- Result: `{"code": "offer_rejected", "message": "Rejected offer o_8.", "ok": true}`
- Reward: `0.008`

## After Training

- Policy: `after_trained_selector:rule_based_expert`
- Task: `coalition_market`
- Seed: `5`
- Episode reward: `1.3412`

### Step 1

- Action: `{"action_type": "accept_offer", "offer_id": "o_1"}`
- Result: `{"code": "offer_accepted", "message": "Accepted offer o_1.", "ok": true}`
- Reward: `0.1097`

### Step 2

- Action: `{"action_type": "reject_offer", "offer_id": "o_2"}`
- Result: `{"code": "offer_rejected", "message": "Rejected offer o_2.", "ok": true}`
- Reward: `0.008`

### Step 3

- Action: `{"action_type": "reject_offer", "offer_id": "o_3"}`
- Result: `{"code": "offer_rejected", "message": "Rejected offer o_3.", "ok": true}`
- Reward: `0.008`

### Step 4

- Action: `{"action_type": "accept_offer", "offer_id": "o_7"}`
- Result: `{"code": "offer_accepted", "message": "Accepted offer o_7.", "ok": true}`
- Reward: `0.1205`

### Step 5

- Action: `{"action_type": "allocate_to_job", "block_ids": ["b_0_1", "b_1_0"], "job_id": "j_0_1"}`
- Result: `{"code": "job_allocated", "message": "Allocated capacity to j_0_1.", "ok": true}`
- Reward: `0.191`

### Step 6

- Action: `{"action_type": "allocate_to_job", "block_ids": ["b_4_0"], "job_id": "j_0_2"}`
- Result: `{"code": "job_allocated", "message": "Allocated capacity to j_0_2.", "ok": true}`
- Reward: `0.191`

### Step 7

- Action: `{"action_type": "form_coalition", "message": "shared capacity for urgent deadlines", "target_lab_id": "lab_2"}`
- Result: `{"code": "coalition_created", "message": "Created coalition c_1.", "ok": true}`
- Reward: `0.128`

### Step 8

- Action: `{"action_type": "accept_offer", "offer_id": "o_8"}`
- Result: `{"code": "offer_accepted", "message": "Accepted offer o_8.", "ok": true}`
- Reward: `0.166`
