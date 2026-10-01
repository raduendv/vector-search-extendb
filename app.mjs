import { readFileSync } from "node:fs";
import { Agent } from "node:https";
import {
  CreateTableCommand,
  DeleteTableCommand,
  DescribeTableCommand,
  DynamoDBClient,
  PutItemCommand,
  SearchVectorsCommand,
} from "@aws-sdk/client-dynamodb";
import { NodeHttpHandler } from "@smithy/node-http-handler";

const required = (name) => {
  const value = process.env[name];
  if (!value) throw new Error(`Missing required environment variable: ${name}`);
  return value;
};

const endpoint = process.env.EXTENDB_ENDPOINT
  ?? process.env.AWS_ENDPOINT_URL_DYNAMODB
  ?? process.env.AWS_ENDPOINT_URL;
if (!endpoint) throw new Error("Missing EXTENDB_ENDPOINT or AWS_ENDPOINT_URL_DYNAMODB");
const region = process.env.AWS_REGION ?? process.env.AWS_DEFAULT_REGION ?? "us-east-1";
const caBundlePath = endpoint.startsWith("https:") ? required("AWS_CA_BUNDLE") : undefined;
const tableName = process.env.TABLE_NAME ?? `vector-search-test-${Date.now()}`;
const indexName = "EmbeddingIndex";
const dimensions = 3;
const pollIntervalMs = 500;
const timeoutMs = 60_000;

const client = new DynamoDBClient({
  endpoint,
  region,
  credentials: {
    accessKeyId: required("AWS_ACCESS_KEY_ID"),
    secretAccessKey: required("AWS_SECRET_ACCESS_KEY"),
  },
  requestHandler: caBundlePath
    ? new NodeHttpHandler({ httpsAgent: new Agent({ ca: readFileSync(caBundlePath) }) })
    : undefined,
});

const vector = (values) => ({ L: values.map((value) => ({ N: String(value) })) });
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function waitUntil(description, check) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await check()) return;
    await sleep(pollIntervalMs);
  }
  throw new Error(`Timed out waiting for ${description}`);
}

async function main() {
  let created = false;
  try {
    console.log(`Connecting to ${endpoint}`);
    console.log(`Creating ${tableName} with vector index ${indexName}`);

    await client.send(new CreateTableCommand({
      TableName: tableName,
      BillingMode: "PAY_PER_REQUEST",
      AttributeDefinitions: [{ AttributeName: "id", AttributeType: "S" }],
      KeySchema: [{ AttributeName: "id", KeyType: "HASH" }],
      VectorIndexes: [{
        IndexName: indexName,
        VectorAttribute: { AttributeName: "embedding" },
        Projection: { ProjectionType: "ALL" },
        Dimensions: dimensions,
        DistanceFunction: "EUCLIDEAN",
      }],
    }));
    created = true;

    await waitUntil("the vector index to become active", async () => {
      const result = await client.send(new DescribeTableCommand({ TableName: tableName }));
      const index = result.Table?.VectorIndexes?.find(({ IndexName }) => IndexName === indexName);
      return result.Table?.TableStatus === "ACTIVE" && index?.IndexStatus === "ACTIVE" && !index.Backfilling;
    });

    const fixtures = [
      { id: "nearest", label: "Expected nearest match", embedding: [1, 0, 0] },
      { id: "middle", label: "Second-nearest match", embedding: [0.7, 0.7, 0] },
      { id: "far", label: "Distant match", embedding: [0, 1, 0] },
    ];

    for (const item of fixtures) {
      await client.send(new PutItemCommand({
        TableName: tableName,
        Item: {
          id: { S: item.id },
          label: { S: item.label },
          embedding: vector(item.embedding),
        },
      }));
    }

    let response;
    await waitUntil("written vectors to become searchable", async () => {
      try {
        response = await client.send(new SearchVectorsCommand({
          TableName: tableName,
          IndexName: indexName,
          SearchVector: [1, 0.05, 0].map((value) => ({ N: String(value) })),
          TopK: 3,
          ProjectionExpression: "id, label",
        }));
        return response.SearchResults?.length === fixtures.length;
      } catch (error) {
        if (error.name === "ValidationException" || error.name === "ResourceNotFoundException") return false;
        throw error;
      }
    });

    const matches = response.SearchResults.map(({ Item, Score }) => ({
      id: Item.id.S,
      label: Item.label.S,
      score: Number(Score),
    }));

    console.table(matches);
    if (matches[0]?.id !== "nearest") {
      throw new Error(`Vector search returned ${matches[0]?.id ?? "no result"} instead of nearest`);
    }
    if (!matches.every((match, index) => index === 0 || matches[index - 1].score <= match.score)) {
      throw new Error("EUCLIDEAN results were not ordered from lowest to highest distance");
    }

    console.log("PASS: ExtendDB vector search returned the expected nearest neighbor and ranking.");
  } finally {
    if (created && process.env.CLEANUP === "true") {
      await client.send(new DeleteTableCommand({ TableName: tableName }));
      console.log(`Deleted ${tableName}`);
    } else if (created) {
      console.log(`Table retained for inspection: ${tableName} (set CLEANUP=true to delete it after the test)`);
    }
    client.destroy();
  }
}

main().catch((error) => {
  console.error("FAIL:", error);
  process.exitCode = 1;
});
