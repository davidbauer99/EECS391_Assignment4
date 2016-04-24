package edu.cwru.sepia.agent;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.action.ActionFeedback;
import edu.cwru.sepia.action.ActionResult;
import edu.cwru.sepia.environment.model.history.DeathLog;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.history.History.HistoryView;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.State.StateView;
import edu.cwru.sepia.environment.model.state.Unit;

public class RLAgent extends Agent {

	/**
	 * Set in the constructor. Defines how many learning episodes your agent should run for.
	 * When starting an episode. If the count is greater than this value print a message
	 * and call sys.exit(0)
	 */
	public final int numEpisodes;

	// tells whether the episodes currently being played are testing or not
	private boolean isTesting = false;
	// the number of non-test episodes completed
	private int episodesPlayed = 0;
	// the number of test episodes completed for the current round of testing
	private int testEpisodesPlayed = 0;
	// the cumulative reward for the current round of testing
	private double cumulativeTestReward = 0.0;
	// A list of average rewards for each round of testing
	private final List<Double> testRewards = new ArrayList<Double>();
	// Rewards for each footman since the last event
	private final Map<Integer, Double> cumulativeRewards = new HashMap<Integer, Double>();
	// Map of enemy footman id to a list of footmen attacking it
	private final Map<Integer, List<Integer>> attackMap = new HashMap<Integer, List<Integer>>();

	/**
	 * List of your footmen and your enemies footmen
	 */
	private List<Integer> myFootmen;
	private List<Integer> enemyFootmen;

	/**
	 * Convenience variable specifying enemy agent number. Use this whenever referring
	 * to the enemy agent. We will make sure it is set to the proper number when testing your code.
	 */
	public static final int ENEMY_PLAYERNUM = 1;

	/**
	 * Set this to whatever size your feature vector is.
	 */
	public static final int NUM_FEATURES = 4;

	/** Use this random number generator for your epsilon exploration. When you submit we will
	 * change this seed so make sure that your agent works for more than the default seed.
	 */
	public final Random random = new Random(12345);

	/**
	 * Your Q-function weights.
	 */
	public Double[] weights;

	/**
	 * These variables are set for you according to the assignment definition. You can change them,
	 * but it is not recommended. If you do change them please let us know and explain your reasoning for
	 * changing them.
	 */
	public final double gamma = 0.9;
	public final double learningRate = .0001;
	public final double epsilon = .02;

	public RLAgent(int playernum, String[] args) {
		super(playernum);

		if (args.length >= 1) {
			numEpisodes = Integer.parseInt(args[0]);
			System.out.println("Running " + numEpisodes + " episodes.");
		} else {
			numEpisodes = 10;
			System.out.println("Warning! Number of episodes not specified. Defaulting to 10 episodes.");
		}

		boolean loadWeights = false;
		if (args.length >= 2) {
			loadWeights = Boolean.parseBoolean(args[1]);
		} else {
			System.out.println("Warning! Load weights argument not specified. Defaulting to not loading.");
		}

		if (loadWeights) {
			weights = loadWeights();
		} else {
			// initialize weights to random values between -1 and 1
			weights = new Double[NUM_FEATURES];
			for (int i = 0; i < weights.length; i++) {
				weights[i] = random.nextDouble() * 2 - 1;
			}
		}
	}

	/**
	 * We've implemented some setup code for your convenience. Change what you need to.
	 */
	@Override
	public Map<Integer, Action> initialStep(State.StateView stateView, History.HistoryView historyView) {

		// You will need to add code to check if you are in a testing or learning episode

		// Find all of your units
		myFootmen = new LinkedList<>();
		for (Integer unitId : stateView.getUnitIds(playernum)) {
			Unit.UnitView unit = stateView.getUnit(unitId);

			String unitName = unit.getTemplateView().getName().toLowerCase();
			if (unitName.equals("footman")) {
				myFootmen.add(unitId);
			} else {
				System.err.println("Unknown unit type: " + unitName);
			}
		}

		// Find all of the enemy units
		enemyFootmen = new LinkedList<>();
		for (Integer unitId : stateView.getUnitIds(ENEMY_PLAYERNUM)) {
			Unit.UnitView unit = stateView.getUnit(unitId);

			String unitName = unit.getTemplateView().getName().toLowerCase();
			if (unitName.equals("footman")) {
				enemyFootmen.add(unitId);
			} else {
				System.err.println("Unknown unit type: " + unitName);
			}
		}

		return middleStep(stateView, historyView);
	}

	/**
	 * You will need to calculate the reward at each step and update your totals. You will also need to
	 * check if an event has occurred. If it has then you will need to update your weights and select a new action.
	 *
	 * If you are using the footmen vectors you will also need to remove killed units. To do so use the historyView
	 * to get a DeathLog. Each DeathLog tells you which player's unit died and the unit ID of the dead unit. To get
	 * the deaths from the last turn do something similar to the following snippet. Please be aware that on the first
	 * turn you should not call this as you will get nothing back.
	 *
	 * for(DeathLog deathLog : historyView.getDeathLogs(stateView.getTurnNumber() -1)) {
	 *     System.out.println("Player: " + deathLog.getController() + " unit: " + deathLog.getDeadUnitID());
	 * }
	 *
	 * You should also check for completed actions using the history view. Obviously you never want a footman just
	 * sitting around doing nothing (the enemy certainly isn't going to stop attacking). So at the minimum you will
	 * have an even whenever one your footmen's targets is killed or an action fails. Actions may fail if the target
	 * is surrounded or the unit cannot find a path to the unit. To get the action results from the previous turn
	 * you can do something similar to the following. Please be aware that on the first turn you should not call this
	 *
	 * Map<Integer, ActionResult> actionResults = historyView.getCommandFeedback(playernum, stateView.getTurnNumber() - 1);
	 * for(ActionResult result : actionResults.values()) {
	 *     System.out.println(result.toString());
	 * }
	 *
	 * @return New actions to execute or nothing if an event has not occurred.
	 */
	@Override
	public Map<Integer, Action> middleStep(State.StateView stateView, History.HistoryView historyView) {
		// handle deaths
		int turnNumber = stateView.getTurnNumber();
		if (0 < turnNumber) {
			for (DeathLog death : historyView.getDeathLogs(turnNumber - 1)) {
				removeDeadUnit(death.getDeadUnitID(), death.getController());
			}
		}

		Map<Integer, Action> actions = new HashMap<Integer, Action>();
		Map<Integer, ActionResult> actResults = null;
		if (0 < turnNumber) {
			actResults = historyView.getCommandFeedback(playernum,
					turnNumber - 1);
		}
		for (int footID : myFootmen) {
			// Reward for this footman
			double reward = calculateReward(stateView, historyView, footID);
			if (cumulativeRewards.get(footID) == null) {
				cumulativeRewards.put(footID, Math.pow(gamma, turnNumber - 1)
						* reward);
			}
			// If testing
			if (isTesting) {
				// Update testing reward
				cumulativeTestReward += Math.pow(gamma, turnNumber - 1) * reward;
			} else if (eventHappened(turnNumber, stateView, historyView)) {
				// If event happened, update the weights
				int targID = getFootmansTarget(footID);
				double[] features = calculateFeatureVector(stateView, historyView, footID, targID);
				weights = updateWeights(weights, features,
						cumulativeRewards.get(footID), stateView,
						historyView, footID);
				// Reset footman's reward since last event to 0
				cumulativeRewards.put(footID, 0.0);
			} else if (0 < turnNumber) {
				// Track footman's reward since last event
				cumulativeRewards.put(footID, cumulativeRewards.get(footID)
						+ Math.pow(gamma, turnNumber - 1) * reward);
			}

			if (needsNewAction(footID, actResults)) {
				// Get the enemy to attack
				int enemyID = selectAction(stateView, historyView, footID);
				updateAttackMap(footID, enemyID);
				actions.put(footID,
						Action.createCompoundAttack(footID, enemyID));
			}
		}
		return actions;
	}

	private void updateAttackMap(int footID, int enemyID) {
		if (attackMap.get(enemyID) == null) {
			attackMap.put(enemyID, new ArrayList<Integer>());
		}
		for (List<Integer> foots : attackMap.values()) {
			if (foots.contains(foots)) {
				foots.remove(foots.indexOf(footID));
			}
		}
		attackMap.get(enemyID).add(footID);
	}

	private boolean needsNewAction(int footID,
			Map<Integer, ActionResult> actResults) {
		if (actResults == null
				|| actResults.get(footID) == null
				|| actResults.get(footID).getFeedback()
				.equals(ActionFeedback.COMPLETED)
				|| actResults.get(footID).getFeedback()
				.equals(ActionFeedback.FAILED)) {
			return true;
		} else {
			return false;
		}
	}

	private int getFootmansTarget(int footID) {
		for (Entry<Integer, List<Integer>> entry : attackMap.entrySet()) {
			if (entry.getValue().contains(footID)) {
				return entry.getKey();
			}
		}
		return 0;
	}

	private boolean eventHappened(int turnNumber, StateView stateView,
			HistoryView historyView) {
		if (0 < turnNumber
				&& (!historyView.getDeathLogs(turnNumber - 1).isEmpty() || !historyView
						.getDamageLogs(turnNumber - 1).isEmpty())) {
			return true;
		} else {
			return false;
		}
	}

	private void removeDeadUnit(int deadUnitID, int controller) {
		if (controller == ENEMY_PLAYERNUM) {
			attackMap.remove(deadUnitID);
			enemyFootmen.remove(enemyFootmen.indexOf(deadUnitID));
		} else {
			myFootmen.remove(myFootmen.indexOf(deadUnitID));
			for (List<Integer> foots : attackMap.values()) {
				if (foots.contains(deadUnitID)) {
					foots.remove(foots.indexOf(deadUnitID));
				}
			}
		}
	}

	/**
	 * Here you will calculate the cumulative average rewards for your testing episodes. If you have just
	 * finished a set of test episodes you will call out testEpisode.
	 *
	 * It is also a good idea to save your weights with the saveWeights function.
	 */
	@Override
	public void terminalStep(State.StateView stateView, History.HistoryView historyView) {
		if (!isTesting) {
			// increment episode number
			episodesPlayed++;
			// If just finished 10, next should be testing
			if (episodesPlayed % 10 == 0) {
				isTesting = true;
			}
			// save the weights
			saveWeights(weights);
		} else {
			// increment test number
			testEpisodesPlayed++;
			// if 5 tests completed
			if (testEpisodesPlayed == 5) {
				// add this round of testing's rewards
				testRewards.add(cumulativeTestReward / testEpisodesPlayed);
				resetTestingState();
				// If we're done, print the test data and exit
				if (episodesPlayed == numEpisodes) {
					printTestData(testRewards);
					System.exit(0);
				}
			}
		}
	}

	private void resetTestingState() {
		isTesting = false;
		testEpisodesPlayed = 0;
		cumulativeTestReward = 0.0;
	}

	/**
	 * Calculate the updated weights for this agent. 
	 * @param oldWeights Weights prior to update
	 * @param oldFeatures Features from (s,a)
	 * @param totalReward Cumulative discounted reward for this footman.
	 * @param stateView Current state of the game.
	 * @param historyView History of the game up until this point
	 * @param footmanId The footman we are updating the weights for
	 * @return The updated weight vector.
	 */
	public Double[] updateWeights(Double[] oldWeights, double[] oldFeatures,
			double totalReward, State.StateView stateView,
			History.HistoryView historyView, int footmanId) {
		//where to get R(s,a): testRewards, but with what index?
		//where to get Q(s',a') options?
		//where to get old Q(s,a) value?
		Double[] result = new Double[oldWeights.length];
		System.arraycopy(oldWeights, 0, result, 0, oldWeights.length);
		double maxSuccQ = 0;//initialization for for loop
		double oldQ = 0;//the old value of Q(s,a)
		for(int i = 0; i < oldWeights.length; i++){
			oldQ = 0;//change this, to be replaced with actual value of Q(s,a)
			maxSuccQ = 0;//change this
			result[i] += learningRate*(testRewards[/*What index??*/] + gamma*maxSuccQ - oldQ)*oldFeatures[i];//update each individual weight
		}
		return result;
	}

	/**
	 * Given a footman and the current state and history of the game select the enemy that this unit should
	 * attack. This is where you would do the epsilon-greedy action selection.
	 *
	 * @param stateView Current state of the game
	 * @param historyView The entire history of this episode
	 * @param attackerId The footman that will be attacking
	 * @return The enemy footman ID this unit should attack
	 */
	public int selectAction(State.StateView stateView, History.HistoryView historyView, int attackerId) {
		int targetID = getHighestQEnemy(stateView, historyView, attackerId);
		// epsilon greedy
		if (this.random.nextDouble() < this.epsilon) {
			int optEnemy = targetID;
			// Temporarily remove best option
			this.enemyFootmen.remove(this.enemyFootmen.indexOf(targetID));
			// get random target
			targetID = this.enemyFootmen.get(this.random
					.nextInt(this.enemyFootmen.size()));
			// replace best option
			this.enemyFootmen.add(optEnemy);
		}
		return targetID;
	}

	// Returns the id of the enemy footman that has the highest Q value
	private int getHighestQEnemy(State.StateView stateView,
			History.HistoryView historyView, int attackerId) {
		double maxValue = Double.NEGATIVE_INFINITY;
		int targetID = -1;
		for (int enemyID : this.enemyFootmen) {
			double value = this.calcQValue(stateView, historyView, attackerId,
					enemyID);
			if (value > maxValue) {
				maxValue = value;
				targetID = enemyID;
			}
		}
		return targetID;
	}

	/**
	 * Given the current state and the footman in question calculate the reward received on the last turn.
	 * This is where you will check for things like Did this footman take or give damage? Did this footman die
	 * or kill its enemy. Did this footman start an action on the last turn? See the assignment description
	 * for the full list of rewards.
	 *
	 * Remember that you will need to discount this reward based on the timestep it is received on. See
	 * the assignment description for more details.
	 *
	 * As part of the reward you will need to calculate if any of the units have taken damage. You can use
	 * the history view to get a list of damages dealt in the previous turn. Use something like the following.
	 *
	 * for(DamageLog damageLogs : historyView.getDamageLogs(lastTurnNumber)) {
	 *     System.out.println("Defending player: " + damageLog.getDefenderController() + " defending unit: " + \
	 *     damageLog.getDefenderID() + " attacking player: " + damageLog.getAttackerController() + \
	 *     "attacking unit: " + damageLog.getAttackerID());
	 * }
	 *
	 * You will do something similar for the deaths. See the middle step documentation for a snippet
	 * showing how to use the deathLogs.
	 *
	 * To see if a command was issued you can check the commands issued log.
	 *
	 * Map<Integer, Action> commandsIssued = historyView.getCommandsIssued(playernum, lastTurnNumber);
	 * for (Map.Entry<Integer, Action> commandEntry : commandsIssued.entrySet()) {
	 *     System.out.println("Unit " + commandEntry.getKey() + " was command to " + commandEntry.getValue().toString);
	 * }
	 *
	 * @param stateView The current state of the game.
	 * @param historyView History of the episode up until this turn.
	 * @param footmanId The footman ID you are looking for the reward from.
	 * @return The current reward
	 */
	public double calculateReward(State.StateView stateView, History.HistoryView historyView, int footmanId) {
		double result = 0;
		boolean actionStartedLastTurn = false;//change this to see if an action was started last turn
		if(actionStartedLastTurn){
			result -= 0.1;
		}
		double discountedDamageDoneToEnemies = 0;//replace with the amt. of damage this footman dealt to enemies
		//make sure the above variable discounts damages when calculating
		//i.e. damage dealt n timesteps after the start of the event is weighted by gamma^n
		result += discountedDamageDoneToEnemies;
		double discountedDamageTaken = 0;//similar to above, but now with damage taken
		result -= discountedDamageTaken;
		double discountedEnemyDeaths = 0;// if an enemy died, equals 100*gamma^(timestep)
		result += discountedEnemyDeaths;
		double discountedAgentDeath = 0;// similar but with the agent dyaing
		result -= discountedAgentDeath;
		return result;
	}

	/**
	 * Calculate the Q-Value for a given state action pair. The state in this scenario is the current
	 * state view and the history of this episode. The action is the attacker and the enemy pair for the
	 * SEPIA attack action.
	 *
	 * This returns the Q-value according to your feature approximation. This is where you will calculate
	 * your features and multiply them by your current weights to get the approximate Q-value.
	 *
	 * @param stateView Current SEPIA state
	 * @param historyView Episode history up to this point in the game
	 * @param attackerId Your footman. The one doing the attacking.
	 * @param defenderId An enemy footman that your footman would be attacking
	 * @return The approximate Q-value
	 */
	public double calcQValue(State.StateView stateView,
			History.HistoryView historyView,
			int attackerId,
			int defenderId) {
		//take a dot product of features array with the weights array
		return 0;
	}

	/**
	 * Given a state and action calculate your features here. Please include a comment explaining what features
	 * you chose and why you chose them.
	 *
	 * All of your feature functions should evaluate to a double. Collect all of these into an array. You will
	 * take a dot product of this array with the weights array to get a Q-value for a given state action.
	 *
	 * It is a good idea to make the first value in your array a constant. This just helps remove any offset
	 * from 0 in the Q-function. The other features are up to you. Many are suggested in the assignment
	 * description.
	 *
	 * @param stateView Current state of the SEPIA game
	 * @param historyView History of the game up until this turn
	 * @param attackerId Your footman. The one doing the attacking.
	 * @param defenderId An enemy footman. The one you are considering attacking.
	 * @return The array of feature function outputs.
	 */
	public double[] calculateFeatureVector(State.StateView stateView,
			History.HistoryView historyView,
			int attackerId,
			int defenderId) {
		double[] result = new double[NUM_FEATURES];
		double attackerHealth = 0;//change this, the HP of the attacker
		double targetHealth = 0;//change this, the HP of the target
		int eTargetedBy = 0;//change this, the number of friendly units attacking e, the target
		int targetDist = 0;//change this, the manhattan distance from the attacker to the target
		double averageOtherDist = 0;//change this, the average distance between the attacker and the enemies other than e
		//feature 0: average manhattan distance to enemies other than e: making this positively weighted should prevent taking a lot of damage
		result[0] = averageOtherDist;
		//feature 1: (target health / how many other units are attacking e): making this positively weighted should prioritize decreasing number of enemies quickly
		result[1] = (targetHealth / eTargetedBy);
		//feature 2: manhattan distance to e: making this negatively weighted should prioritize reaching the enemy quickly
		result[2] = targetDist;
		//feature 3: attacker health - target health
		result[3] = attackerHealth -targetHealth;
		return result;
	}

	/**
	 * DO NOT CHANGE THIS!
	 *
	 * Prints the learning rate data described in the assignment. Do not modify this method.
	 *
	 * @param averageRewards List of cumulative average rewards from test episodes.
	 */
	public void printTestData (List<Double> averageRewards) {
		System.out.println("");
		System.out.println("Games Played      Average Cumulative Reward");
		System.out.println("-------------     -------------------------");
		for (int i = 0; i < averageRewards.size(); i++) {
			String gamesPlayed = Integer.toString(10*i);
			String averageReward = String.format("%.2f", averageRewards.get(i));

			int numSpaces = "-------------     ".length() - gamesPlayed.length();
			StringBuffer spaceBuffer = new StringBuffer(numSpaces);
			for (int j = 0; j < numSpaces; j++) {
				spaceBuffer.append(" ");
			}
			System.out.println(gamesPlayed + spaceBuffer.toString() + averageReward);
		}
		System.out.println("");
	}

	/**
	 * DO NOT CHANGE THIS!
	 *
	 * This function will take your set of weights and save them to a file. Overwriting whatever file is
	 * currently there. You will use this when training your agents. You will include th output of this function
	 * from your trained agent with your submission.
	 *
	 * Look in the agent_weights folder for the output.
	 *
	 * @param weights Array of weights
	 */
	public void saveWeights(Double[] weights) {
		File path = new File("agent_weights/weights.txt");
		// create the directories if they do not already exist
		path.getAbsoluteFile().getParentFile().mkdirs();

		try {
			// open a new file writer. Set append to false
			BufferedWriter writer = new BufferedWriter(new FileWriter(path, false));

			for (double weight : weights) {
				writer.write(String.format("%f\n", weight));
			}
			writer.flush();
			writer.close();
		} catch(IOException ex) {
			System.err.println("Failed to write weights to file. Reason: " + ex.getMessage());
		}
	}

	/**
	 * DO NOT CHANGE THIS!
	 *
	 * This function will load the weights stored at agent_weights/weights.txt. The contents of this file
	 * can be created using the saveWeights function. You will use this function if the load weights argument
	 * of the agent is set to 1.
	 *
	 * @return The array of weights
	 */
	public Double[] loadWeights() {
		File path = new File("agent_weights/weights.txt");
		if (!path.exists()) {
			System.err.println("Failed to load weights. File does not exist");
			return null;
		}

		try {
			BufferedReader reader = new BufferedReader(new FileReader(path));
			String line;
			List<Double> weights = new LinkedList<>();
			while((line = reader.readLine()) != null) {
				weights.add(Double.parseDouble(line));
			}
			reader.close();

			return weights.toArray(new Double[weights.size()]);
		} catch(IOException ex) {
			System.err.println("Failed to load weights from file. Reason: " + ex.getMessage());
		}
		return null;
	}

	@Override
	public void savePlayerData(OutputStream outputStream) {

	}

	@Override
	public void loadPlayerData(InputStream inputStream) {

	}
}
