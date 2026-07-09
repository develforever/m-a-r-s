/**
 * Mars Terraform Game State
 * 
 * The NPC assistant analyzes this state vector every frame
 * using the M.A.R.S. SOM-Router to decide which strategy to activate.
 */

export interface Resource {
  name: string;
  amount: number;
  maxAmount: number;
  rate: number; // per second
  color: string;
}

export interface GameState {
  resources: Resource[];
  time: number;
  activePod: number;
  confidence: number;
  routingTimeUs: number;
  frameCount: number;
}

export const STRATEGIES = [
  { id: 0, name: 'Water Extraction', icon: '💧', color: '#3b82f6' },
  { id: 1, name: 'Mineral Mining', icon: '⛏️', color: '#f59e0b' },
  { id: 2, name: 'Energy Production', icon: '⚡', color: '#10b981' },
  { id: 3, name: 'Atmosphere Processing', icon: '🌬️', color: '#8b5cf6' },
];

export function createInitialState(): GameState {
  return {
    resources: [
      { name: 'Water', amount: 30, maxAmount: 100, rate: 0, color: '#3b82f6' },
      { name: 'Minerals', amount: 50, maxAmount: 100, rate: 0, color: '#f59e0b' },
      { name: 'Energy', amount: 70, maxAmount: 100, rate: 0, color: '#10b981' },
      { name: 'Oxygen', amount: 20, maxAmount: 100, rate: 0, color: '#8b5cf6' },
    ],
    time: 0,
    activePod: 0,
    confidence: 0,
    routingTimeUs: 0,
    frameCount: 0,
  };
}

/**
 * Encode game state into a flat vector for the SOM-Router.
 * Dimensionality: N_IN = 16 (4 resources × 4 features each)
 */
export function encodeGameState(state: GameState): Float32Array {
  const vec = new Float32Array(16);
  
  for (let i = 0; i < 4; i++) {
    const r = state.resources[i];
    vec[i * 4 + 0] = r.amount / r.maxAmount;        // normalized amount
    vec[i * 4 + 1] = r.rate;                        // production rate
    vec[i * 4 + 2] = (r.maxAmount - r.amount) / r.maxAmount; // deficit
    vec[i * 4 + 3] = Math.sin(state.time * 0.1 + i); // temporal pattern
  }
  
  return vec;
}

/**
 * Update game state based on active strategy (pod).
 * Each strategy boosts one resource at the cost of energy.
 */
export function updateGameState(state: GameState, dt: number): GameState {
  const newState = { ...state, resources: state.resources.map(r => ({ ...r })) };
  newState.time += dt;
  newState.frameCount++;

  // Natural decay
  for (const r of newState.resources) {
    r.amount = Math.max(0, r.amount - 0.3 * dt);
    r.rate = 0;
  }

  // Active strategy produces its resource
  const pod = state.activePod;
  if (pod >= 0 && pod < 4) {
    const boost = 2.0 * dt * state.confidence;
    newState.resources[pod].amount = Math.min(
      newState.resources[pod].maxAmount,
      newState.resources[pod].amount + boost
    );
    newState.resources[pod].rate = boost / dt;

    // Energy cost (unless producing energy)
    if (pod !== 2) {
      newState.resources[2].amount = Math.max(
        0, newState.resources[2].amount - 0.5 * dt
      );
    }
  }

  return newState;
}

/**
 * Simple CPU heuristic router (baseline comparison).
 * Returns the index of the resource with the lowest amount.
 */
export function heuristicRoute(state: GameState): number {
  let minIdx = 0;
  let minAmount = Infinity;
  for (let i = 0; i < state.resources.length; i++) {
    if (state.resources[i].amount < minAmount) {
      minAmount = state.resources[i].amount;
      minIdx = i;
    }
  }
  return minIdx;
}
